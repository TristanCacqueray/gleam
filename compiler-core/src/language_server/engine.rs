use std::sync::Arc;

use crate::{
    ast::{
        Arg, Definition, Function, Import, ModuleConstant, TypedDefinition, TypedExpr, TypedPattern,
    },
    build::{Located, Module},
    config::PackageConfig,
    io::{CommandExecutor, FileSystemReader, FileSystemWriter},
    language_server::{
        compiler::LspProjectCompiler, files::FileSystemProxy, progress::ProgressReporter, SrcSpan,
    },
    line_numbers::LineNumbers,
    paths::ProjectPaths,
    type_::{pretty::Printer, PreludeType, Type, ValueConstructorVariant},
    Error, Result, Warning,
};
use camino::Utf8PathBuf;
use lsp_types::{self as lsp, Hover, HoverContents, MarkedString, Url};
use smol_str::SmolStr;
use std::collections::HashMap;
use strum::IntoEnumIterator;

use super::{src_span_to_lsp_range, DownloadDependencies, MakeLocker};

#[derive(Debug, PartialEq, Eq)]
pub struct Response<T> {
    pub result: Result<T, Error>,
    pub warnings: Vec<Warning>,
    pub compilation: Compilation,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Compilation {
    /// Compilation was attempted and succeeded for these modules.
    Yes(Vec<Utf8PathBuf>),
    /// Compilation was not attempted for this operation.
    No,
}

#[derive(Debug)]
pub struct LanguageServerEngine<IO, Reporter> {
    pub(crate) paths: ProjectPaths,

    /// A compiler for the project that supports repeat compilation of the root
    /// package.
    /// In the event the the project config changes this will need to be
    /// discarded and reloaded to handle any changes to dependencies.
    pub(crate) compiler: LspProjectCompiler<FileSystemProxy<IO>>,

    modules_compiled_since_last_feedback: Vec<Utf8PathBuf>,
    compiled_since_last_feedback: bool,

    // Used to publish progress notifications to the client without waiting for
    // the usual request-response loop.
    progress_reporter: Reporter,

    /// The storage for unused warning location.
    unused_warnings_locations: HashMap<Utf8PathBuf, Vec<lsp_types::Range>>,
}

impl<'a, IO, Reporter> LanguageServerEngine<IO, Reporter>
where
    // IO to be supplied from outside of gleam-core
    IO: FileSystemReader
        + FileSystemWriter
        + CommandExecutor
        + DownloadDependencies
        + MakeLocker
        + Clone,
    // IO to be supplied from inside of gleam-core
    Reporter: ProgressReporter + Clone + 'a,
{
    pub fn new(
        config: PackageConfig,
        progress_reporter: Reporter,
        io: FileSystemProxy<IO>,
        paths: ProjectPaths,
    ) -> Result<Self> {
        let locker = io.inner().make_locker(&paths, config.target)?;

        // Download dependencies to ensure they are up-to-date for this new
        // configuration and new instance of the compiler
        progress_reporter.dependency_downloading_started();
        let manifest = io.inner().download_dependencies(&paths);
        progress_reporter.dependency_downloading_finished();

        // NOTE: This must come after the progress reporter has finished!
        let manifest = manifest?;

        let compiler =
            LspProjectCompiler::new(manifest, config, paths.clone(), io.clone(), locker)?;

        Ok(Self {
            modules_compiled_since_last_feedback: vec![],
            compiled_since_last_feedback: false,
            unused_warnings_locations: HashMap::new(),
            progress_reporter,
            compiler,
            paths,
        })
    }

    pub fn compile_please(&mut self) -> Response<()> {
        self.respond(Self::compile)
    }

    /// Compile the project if we are in one. Otherwise do nothing.
    fn compile(&mut self) -> Result<(), Error> {
        self.compiled_since_last_feedback = true;

        self.progress_reporter.compilation_started();
        let result = self.compiler.compile();
        self.progress_reporter.compilation_finished();

        let modules = result?;
        self.modules_compiled_since_last_feedback.extend(modules);

        Ok(())
    }

    fn take_warnings(&mut self) -> Vec<Warning> {
        self.compiler.take_warnings()
    }

    // TODO: test local variables
    // TODO: test same module constants
    // TODO: test imported module constants
    // TODO: test unqualified imported module constants
    // TODO: test same module records
    // TODO: test imported module records
    // TODO: test unqualified imported module records
    // TODO: test same module functions
    // TODO: test module function calls
    // TODO: test different package module function calls
    //
    //
    //
    // TODO: implement unqualified imported module functions
    // TODO: implement goto definition of modules that do not belong to the top
    // level package.
    //
    pub fn goto_definition(
        &mut self,
        params: lsp::GotoDefinitionParams,
    ) -> Response<Option<lsp::Location>> {
        self.respond(|this| {
            let params = params.text_document_position_params;
            let (line_numbers, node) = match this.node_at_position(&params) {
                Some(location) => location,
                None => return Ok(None),
            };

            let location = match node.definition_location() {
                Some(location) => location,
                None => return Ok(None),
            };

            let (uri, line_numbers) = match location.module {
                None => (params.text_document.uri, &line_numbers),
                Some(name) => {
                    let module = match this.compiler.get_source(name) {
                        Some(module) => module,
                        // TODO: support goto definition for functions defined in
                        // different packages. Currently it is not possible as the
                        // required LineNumbers and source file path information is
                        // not stored in the module metadata.
                        None => return Ok(None),
                    };
                    let url = Url::parse(&format!("file:///{}", &module.path))
                        .expect("goto definition URL parse");
                    (url, &module.line_numbers)
                }
            };
            let range = src_span_to_lsp_range(location.span, line_numbers);

            Ok(Some(lsp::Location { uri, range }))
        })
    }

    pub fn completion(
        &mut self,
        params: lsp::TextDocumentPositionParams,
    ) -> Response<Option<Vec<lsp::CompletionItem>>> {
        self.respond(|this| {
            let module = match this.module_for_uri(&params.text_document.uri) {
                Some(m) => m,
                None => return Ok(None),
            };

            let line_numbers = LineNumbers::new(&module.code);
            let byte_index =
                line_numbers.byte_index(params.position.line, params.position.character);

            let Some(found) = module.find_node(byte_index) else {
                return Ok(None);
            };

            let completions = match found {
                Located::Pattern(_pattern) => None,

                Located::Statement(_) | Located::Expression(_) => {
                    Some(this.completion_values(module))
                }

                Located::ModuleStatement(Definition::Function(_)) => {
                    Some(this.completion_types(module))
                }

                Located::FunctionBody(_) => Some(this.completion_values(module)),

                Located::ModuleStatement(Definition::TypeAlias(_) | Definition::CustomType(_)) => {
                    Some(this.completion_types(module))
                }

                Located::ModuleStatement(Definition::Import(_) | Definition::ModuleConstant(_)) => {
                    None
                }

                Located::Arg(_) => None,
            };

            Ok(completions)
        })
    }

    pub fn action(
        &mut self,
        params: lsp::CodeActionParams,
    ) -> Response<Option<Vec<lsp_types::CodeAction>>> {
        self.respond(|this| {
            let mut actions = vec![];

            // Check if unused removal can be performed
            if is_action_attached_to_unused_diagnostic(&params) {
                let path: Utf8PathBuf = params.text_document.uri.path().into();
                if let Some(ranges) = this.unused_warnings_locations.get(&path) {
                    // Unused ranges were previously computed, offer a new code action:
                    actions.push(make_unused_code_action(params.text_document.uri, ranges))
                }
            }

            Ok(if actions.is_empty() {
                None
            } else {
                Some(actions)
            })
        })
    }

    fn module_code(&self, path: &Utf8PathBuf) -> Option<(LineNumbers, &'_ Module, SmolStr)> {
        let uri = crate::language_server::server::path_to_uri(path.clone());
        let module = self.module_for_uri(&uri)?;
        let line_numbers = LineNumbers::new(&module.code);
        Some((line_numbers, module, module.code.clone()))
    }

    // This function remember unused warning locations for the related code action.
    fn store_unused_warnings(&mut self, modules: &[Utf8PathBuf], warnings: &[Warning]) {
        // Remove previous locations of the newly compiled module
        for module in modules {
            let _ = self.unused_warnings_locations.remove(module);
        }

        // Group warnings per module
        let mut warnings_per_module: HashMap<Utf8PathBuf, Vec<&crate::type_::Warning>> =
            HashMap::new();
        for warning in warnings {
            if let Warning::Type { path, warning, .. } = warning {
                if let Some(vec) = warnings_per_module.get_mut(path) {
                    vec.push(warning);
                } else {
                    let _ = warnings_per_module.insert(path.clone(), vec![warning]);
                }
            }
        }

        let mut ranges_per_location: HashMap<Utf8PathBuf, Vec<lsp_types::Range>> = HashMap::new();

        // Record unused locations
        for (path, warnings) in warnings_per_module.drain() {
            if let Some((line_numbers, module, code)) = self.module_code(&path) {
                let mut qual_per_module: HashMap<SmolStr, QualifiedUnusedInfo> = HashMap::new();
                for warning in warnings {
                    match warning {
                        crate::type_::Warning::UnusedImportedModule { location, .. } => {
                            if let Some(range) =
                                handle_unused_imported_module(&line_numbers, &code, location)
                            {
                                store_unused_warning_range(&mut ranges_per_location, &path, range);
                            }
                        }

                        crate::type_::Warning::UnusedImportedValue { location, .. } => {
                            if let Some(Located::ModuleStatement(
                                crate::language_server::engine::Definition::Import(import),
                            )) = module.find_node(location.start)
                            {
                                // Store the qualified import details now, we'll process it after.
                                match qual_per_module.get_mut(&import.module) {
                                    Some(info) => info.unused.push(*location),
                                    None => {
                                        let _ = qual_per_module.insert(
                                            import.module.clone(),
                                            QualifiedUnusedInfo {
                                                unqualified_count: import.unqualified.len(),
                                                has_name: import.as_name.is_some(),
                                                location: import.location,
                                                unused: vec![*location],
                                            },
                                        );
                                    }
                                }
                            };
                        }
                        _ => (),
                    }
                }

                for (_, info) in qual_per_module.drain() {
                    if let Some(ranges) =
                        handle_unused_imported_module_qualified(&line_numbers, &code, &info)
                    {
                        for range in ranges {
                            store_unused_warning_range(&mut ranges_per_location, &path, range);
                        }
                    }
                }
            }
        }

        // Copy all the ranges to the engine.
        for (path, ranges) in ranges_per_location.drain() {
            let _ = self.unused_warnings_locations.insert(path, ranges);
        }
    }

    fn respond<T>(&mut self, handler: impl FnOnce(&mut Self) -> Result<T>) -> Response<T> {
        let result = handler(self);
        let warnings = self.take_warnings();
        // TODO: test. Ensure hover doesn't report as compiled
        let compilation = if self.compiled_since_last_feedback {
            let modules = std::mem::take(&mut self.modules_compiled_since_last_feedback);
            self.compiled_since_last_feedback = false;
            self.store_unused_warnings(&modules, &warnings);
            Compilation::Yes(modules)
        } else {
            Compilation::No
        };
        Response {
            result,
            warnings,
            compilation,
        }
    }

    pub fn hover(&mut self, params: lsp::HoverParams) -> Response<Option<Hover>> {
        self.respond(|this| {
            let params = params.text_document_position_params;

            let (lines, found) = match this.node_at_position(&params) {
                Some(value) => value,
                None => return Ok(None),
            };

            Ok(match found {
                Located::Statement(_) => None, // TODO: hover for statement
                Located::ModuleStatement(Definition::Function(fun)) => {
                    Some(hover_for_function_head(fun, lines))
                }
                Located::ModuleStatement(Definition::ModuleConstant(constant)) => {
                    Some(hover_for_module_constant(constant, lines))
                }
                Located::ModuleStatement(_) => None,
                Located::Pattern(pattern) => Some(hover_for_pattern(pattern, lines)),
                Located::Expression(expression) => Some(hover_for_expression(expression, lines)),
                Located::Arg(arg) => Some(hover_for_function_argument(arg, lines)),
                Located::FunctionBody(_) => None,
            })
        })
    }

    fn module_node_at_position(
        &self,
        params: &lsp::TextDocumentPositionParams,
        module: &'a Module,
    ) -> Option<(LineNumbers, Located<'a>)> {
        let line_numbers = LineNumbers::new(&module.code);
        let byte_index = line_numbers.byte_index(params.position.line, params.position.character);
        let node = module.find_node(byte_index);
        let node = node?;
        Some((line_numbers, node))
    }

    fn node_at_position(
        &self,
        params: &lsp::TextDocumentPositionParams,
    ) -> Option<(LineNumbers, Located<'_>)> {
        let module = self.module_for_uri(&params.text_document.uri)?;
        self.module_node_at_position(params, module)
    }

    fn module_for_uri(&self, uri: &Url) -> Option<&Module> {
        use itertools::Itertools;

        // The to_file_path method is available on these platforms
        #[cfg(any(unix, windows, target_os = "redox", target_os = "wasi"))]
        let path = uri.to_file_path().expect("URL file");

        #[cfg(not(any(unix, windows, target_os = "redox", target_os = "wasi")))]
        let path: Utf8PathBuf = uri.path().into();

        let components = path
            .strip_prefix(self.paths.root())
            .ok()?
            .components()
            .skip(1)
            .map(|c| c.as_os_str().to_string_lossy());
        let module_name: SmolStr = Itertools::intersperse(components, "/".into())
            .collect::<String>()
            .strip_suffix(".gleam")?
            .into();

        self.compiler.modules.get(&module_name)
    }

    fn completion_types<'b>(&'b self, module: &'b Module) -> Vec<lsp::CompletionItem> {
        let mut completions = vec![];

        // Prelude types
        for type_ in PreludeType::iter() {
            completions.push(lsp::CompletionItem {
                label: type_.name().into(),
                detail: Some("Type".into()),
                kind: Some(lsp::CompletionItemKind::CLASS),
                ..Default::default()
            });
        }

        // Module types
        for (name, type_) in &module.ast.type_info.types {
            completions.push(type_completion(None, name, type_));
        }

        // Imported modules
        for import in module.ast.definitions.iter().filter_map(get_import) {
            let alias = import.used_name();

            // The module may not be known of yet if it has not previously
            // compiled yet in this editor session.
            // TODO: test getting completions from modules defined in other packages
            let Some(module) = self.compiler.get_module_inferface(&import.module) else {
                continue;
            };

            // Qualified types
            for (name, type_) in &module.types {
                if !type_.public {
                    continue;
                }
                completions.push(type_completion(Some(&alias), name, type_));
            }

            // Unqualified types
            for unqualified in &import.unqualified {
                let Some(type_) = module.get_public_type(&unqualified.name) else {
                    continue;
                };
                completions.push(type_completion(None, unqualified.variable_name(), type_));
            }
        }

        completions
    }

    fn completion_values<'b>(&'b self, module: &'b Module) -> Vec<lsp::CompletionItem> {
        let mut completions = vec![];

        // Module functions
        for (name, value) in &module.ast.type_info.values {
            completions.push(value_completion(None, name, value));
        }

        // Imported modules
        for import in module.ast.definitions.iter().filter_map(get_import) {
            let alias = import.used_name();

            // The module may not be known of yet if it has not previously
            // compiled yet in this editor session.
            // TODO: test getting completions from modules defined in other packages
            let Some(module) = self.compiler.get_module_inferface(&import.module) else {
                continue;
            };

            // Qualified values
            for (name, value) in &module.values {
                if !value.public {
                    continue;
                }
                completions.push(value_completion(Some(&alias), name, value));
            }

            // Unqualified values
            for unqualified in &import.unqualified {
                let Some(value) = module.get_public_value(&unqualified.name) else {
                    continue;
                };
                completions.push(value_completion(None, unqualified.variable_name(), value));
            }
        }

        completions
    }
}

fn type_completion(
    module: Option<&SmolStr>,
    name: &str,
    type_: &crate::type_::TypeConstructor,
) -> lsp::CompletionItem {
    let label = match module {
        Some(module) => format!("{module}.{name}"),
        None => name.to_string(),
    };

    let kind = Some(if type_.typ.is_variable() {
        lsp::CompletionItemKind::VARIABLE
    } else {
        lsp::CompletionItemKind::CLASS
    });

    lsp::CompletionItem {
        label,
        kind,
        detail: Some("Type".into()),
        ..Default::default()
    }
}

fn value_completion(
    module: Option<&str>,
    name: &str,
    value: &crate::type_::ValueConstructor,
) -> lsp::CompletionItem {
    let label = match module {
        Some(module) => format!("{module}.{name}"),
        None => name.to_string(),
    };

    let type_ = Printer::new().pretty_print(&value.type_, 0);

    let kind = Some(match value.variant {
        ValueConstructorVariant::LocalVariable { .. } => lsp::CompletionItemKind::VARIABLE,
        ValueConstructorVariant::ModuleConstant { .. } => lsp::CompletionItemKind::CONSTANT,
        ValueConstructorVariant::ModuleFn { .. } => lsp::CompletionItemKind::FUNCTION,
        ValueConstructorVariant::Record { arity: 0, .. } => lsp::CompletionItemKind::ENUM_MEMBER,
        ValueConstructorVariant::Record { .. } => lsp::CompletionItemKind::CONSTRUCTOR,
    });

    let documentation = value.get_documentation().map(|d| {
        lsp::Documentation::MarkupContent(lsp::MarkupContent {
            kind: lsp::MarkupKind::Markdown,
            value: d.to_string(),
        })
    });

    lsp::CompletionItem {
        label,
        kind,
        detail: Some(type_),
        documentation,
        ..Default::default()
    }
}

fn get_import(statement: &TypedDefinition) -> Option<&Import<SmolStr>> {
    match statement {
        Definition::Import(import) => Some(import),
        _ => None,
    }
}

fn hover_for_pattern(pattern: &TypedPattern, line_numbers: LineNumbers) -> Hover {
    let documentation = pattern.get_documentation().unwrap_or_default();

    // Show the type of the hovered node to the user
    let type_ = Printer::new().pretty_print(pattern.type_().as_ref(), 0);
    let contents = format!(
        "```gleam
{type_}
```
{documentation}"
    );
    Hover {
        contents: HoverContents::Scalar(MarkedString::String(contents)),
        range: Some(src_span_to_lsp_range(pattern.location(), &line_numbers)),
    }
}

fn hover_for_function_head(
    fun: &Function<Arc<Type>, TypedExpr>,
    line_numbers: LineNumbers,
) -> Hover {
    let empty_smolstr = SmolStr::from("");
    let documentation = fun.documentation.as_ref().unwrap_or(&empty_smolstr);
    let function_type = Type::Fn {
        args: fun.arguments.iter().map(|arg| arg.type_.clone()).collect(),
        retrn: fun.return_type.clone(),
    };
    let formatted_type = Printer::new().pretty_print(&function_type, 0);
    let contents = format!(
        "```gleam
{formatted_type}
```
{documentation}"
    );
    Hover {
        contents: HoverContents::Scalar(MarkedString::String(contents)),
        range: Some(src_span_to_lsp_range(fun.location, &line_numbers)),
    }
}

fn hover_for_function_argument(argument: &Arg<Arc<Type>>, line_numbers: LineNumbers) -> Hover {
    let type_ = Printer::new().pretty_print(&argument.type_, 0);
    let contents = format!("```gleam\n{type_}\n```");
    Hover {
        contents: HoverContents::Scalar(MarkedString::String(contents)),
        range: Some(src_span_to_lsp_range(argument.location, &line_numbers)),
    }
}

fn hover_for_module_constant(
    constant: &ModuleConstant<Arc<Type>, SmolStr>,
    line_numbers: LineNumbers,
) -> Hover {
    let empty_smolstr = SmolStr::from("");
    let type_ = Printer::new().pretty_print(&constant.type_, 0);
    let documentation = constant.documentation.as_ref().unwrap_or(&empty_smolstr);
    let contents = format!("```gleam\n{type_}\n```\n{documentation}");
    Hover {
        contents: HoverContents::Scalar(MarkedString::String(contents)),
        range: Some(src_span_to_lsp_range(constant.location, &line_numbers)),
    }
}

fn hover_for_expression(expression: &TypedExpr, line_numbers: LineNumbers) -> Hover {
    let documentation = expression.get_documentation().unwrap_or_default();

    // Show the type of the hovered node to the user
    let type_ = Printer::new().pretty_print(expression.type_().as_ref(), 0);
    let contents = format!(
        "```gleam
{type_}
```
{documentation}"
    );
    Hover {
        contents: HoverContents::Scalar(MarkedString::String(contents)),
        range: Some(src_span_to_lsp_range(expression.location(), &line_numbers)),
    }
}

fn is_action_attached_to_unused_diagnostic(params: &lsp::CodeActionParams) -> bool {
    params
        .context
        .diagnostics
        .iter()
        .any(|diag| diag.message.starts_with("Unused"))
}

fn store_unused_warning_range(
    cache: &mut HashMap<Utf8PathBuf, Vec<lsp_types::Range>>,
    path: &Utf8PathBuf,
    range: lsp_types::Range,
) {
    match cache.get_mut(path) {
        None => {
            let _ = cache.insert(path.clone(), vec![range]);
        }
        Some(vec) => vec.push(range),
    }
}

fn import_token_location(before: &str) -> Option<usize> {
    let pos = before.rfind("import")?;
    // Consume the previous line return if it exists.
    Some(if pos > 0 { pos - 1 } else { pos })
}

fn mk_range(line_numbers: &LineNumbers, start: usize, end: usize) -> lsp_types::Range {
    let location = SrcSpan {
        start: start as u32,
        end: end as u32,
    };
    src_span_to_lsp_range(location, line_numbers)
}

// This function handle UnusedImportedModule warning.
// It finds the full range of the import statement,
// because the warning only contains the location of the module name.
fn handle_unused_imported_module(
    line_numbers: &LineNumbers,
    code: &SmolStr,
    location: &SrcSpan,
) -> Option<lsp_types::Range> {
    let (before, after) = code.split_at(location.start as usize);
    // Find the previous import keyword.
    let start = import_token_location(before)?;
    // Find the next line return.
    let end = location.start as usize + (after.find('\n')?);
    // The src span of the full import statement.
    Some(mk_range(line_numbers, start, end))
}

#[derive(Debug)]
struct QualifiedUnusedInfo {
    unqualified_count: usize,
    has_name: bool,
    location: SrcSpan,
    unused: Vec<SrcSpan>,
}
fn handle_unused_imported_module_qualified(
    line_numbers: &LineNumbers,
    code: &SmolStr,
    info: &QualifiedUnusedInfo,
) -> Option<Vec<lsp_types::Range>> {
    if info.unqualified_count == info.unused.len() {
        // All the qualified import are unused, we can remove the whole import line
        let (before, after) = code.split_at(info.location.start as usize);
        if !info.has_name {
            // Find the previous import keyword.
            let start = import_token_location(before)?;
            // Find the closing '}'.
            let closing_bracket = after.find('}')?;
            // Find the last '\n'
            let (_, rest) = after.split_at(closing_bracket);
            let end = info.location.start as usize + (closing_bracket + rest.find('\n')?);
            Some(vec![mk_range(line_numbers, start, end)])
        } else {
            // Unless the import as a name, then we don't know and we should keep it
            let start = info.location.start as usize + after.find(".{")?;
            let end = info.location.start as usize + 1 + after.find('}')?;
            Some(vec![mk_range(line_numbers, start, end)])
        }
    } else {
        // We can only remove individual elems.
        let mut ranges = Vec::new();
        for span in &info.unused {
            let (before, after) = code.split_at(span.start as usize);
            let (start, end) = match before.rfind(',') {
                Some(comma_loc) if Some(comma_loc) > before.rfind("import") => {
                    // This is not the first qualified name, we drop until the previous comma.
                    (comma_loc, span.end as usize)
                }
                _ => {
                    // This is the first qualified name, we drop until after the next comma.
                    let (_, rest) = after.split_at(after.find(',')?);
                    let whitespaces = rest
                        .chars()
                        .take_while(|c| matches!(c, ',' | '\n' | ' '))
                        .collect::<Vec<_>>()
                        .len();
                    (span.start as usize, span.end as usize + whitespaces)
                }
            };
            ranges.push(mk_range(line_numbers, start, end))
        }
        Some(ranges)
    }
}

// Convert a list of unused range into a "Remove unused" code action.
fn make_unused_code_action(uri: Url, ranges: &[lsp_types::Range]) -> lsp_types::CodeAction {
    use itertools::Itertools;

    let edits = ranges
        .iter()
        .sorted_by(|a, b| Ord::cmp(&a.start, &b.start))
        .map(|range| lsp_types::TextEdit {
            range: *range,
            new_text: "".to_string(),
        })
        .collect();
    let mut changes = std::collections::HashMap::new();
    let _ = changes.insert(uri, edits);
    lsp_types::CodeAction {
        title: "Remove unused imports".to_string(),
        kind: None,
        diagnostics: None,
        edit: Some(lsp_types::WorkspaceEdit {
            changes: Some(changes),
            document_changes: None,
            change_annotations: None,
        }),
        command: None,
        is_preferred: Some(true),
        disabled: None,
        data: None,
    }
}
