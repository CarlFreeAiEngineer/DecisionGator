//! Offline question/context classification through a length-delimited C ABI.
#[cfg(unix)]
use ort::AsPointer;
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::sync::atomic::{AtomicPtr, Ordering};
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::c_char,
    fs,
    io::Read,
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    ptr, slice, str,
    sync::Mutex,
};
use tokenizers::Tokenizer;

type Failure = (i32, String);
type Result<T> = std::result::Result<T, Failure>;
const INVALID: i32 = 1;
const LOAD: i32 = 2;
const INCOMPATIBLE: i32 = 3;
const RESOURCE: i32 = 4;
const INFERENCE: i32 = 5;
const INTERNAL: i32 = 6;
const TOO_LONG: i32 = 7;
const MAX_INPUT_BYTES: usize = 1_048_576;
const MAX_OPTIONS: usize = 256;
static RUNTIME: Mutex<Option<String>> = Mutex::new(None);
static DEFAULT_MODEL: Mutex<Option<GateSession>> = Mutex::new(None);
#[cfg(unix)]
static EXIT_ENVIRONMENT: AtomicPtr<ort::sys::OrtEnv> = AtomicPtr::new(ptr::null_mut());
#[cfg(unix)]
extern "C" {
    fn atexit(callback: extern "C" fn()) -> i32;
}

// ort rc10 keeps its environment in a Rust static, whose destructor never runs.
// Release it before ONNX Runtime destroys its logging mutexes at process exit.
// This matches the lifecycle fix in newer ort versions; do not unload this
// component while the process is running. Ordinary model handles remain freely
// releasable/reloadable. Remove this adapter when upgrading to ort with cleanup.
// Upstream lifecycle: https://docs.rs/ort/latest/src/ort/environment.rs.html
// ONNX issue: https://github.com/microsoft/onnxruntime/issues/24579
// Registration occurs once under RUNTIME's lock; the atomic swap prevents a
// second release. ort rc10's static environment has no other destructor.
// Windows deliberately does NOT register this callback: MSVC runs a DLL's
// atexit callbacks at DLL detach, when ORT may already have detached and worker
// threads have been terminated. Calling ORT or dropping a session then is unsafe.
// Keep the Windows default session/environment for the lifetime of the process;
// the OS reclaims them at termination. Explicit dg_release still drops sessions
// normally during program execution. Dynamic unloading remains unsupported.
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/atexit
// https://learn.microsoft.com/en-us/windows/win32/dlls/dllmain
#[cfg(unix)]
extern "C" fn release_environment_at_exit() {
    // The lazy model is a static too: release its session before its environment.
    if let Ok(mut model) = DEFAULT_MODEL.lock() {
        *model = None;
    }
    let environment = EXIT_ENVIRONMENT.swap(ptr::null_mut(), Ordering::SeqCst);
    if !environment.is_null() {
        unsafe {
            (ort::api().ReleaseEnv)(environment);
        }
    }
}

#[cfg(unix)]
#[cfg_attr(dg_embedded, allow(dead_code))]
fn component_directory() -> Result<PathBuf> {
    use std::ffi::{c_void, CStr};
    use std::os::unix::ffi::OsStrExt;
    #[repr(C)]
    struct DlInfo {
        filename: *const c_char,
        base: *mut c_void,
        symbol: *const c_char,
        address: *mut c_void,
    }
    #[cfg_attr(target_os = "linux", link(name = "dl"))]
    extern "C" {
        fn dladdr(address: *const c_void, info: *mut DlInfo) -> i32;
    }
    let mut info = DlInfo {
        filename: ptr::null(),
        base: ptr::null_mut(),
        symbol: ptr::null(),
        address: ptr::null_mut(),
    };
    let found = unsafe { dladdr(dg_is_yes_p as *const () as *const c_void, &mut info) };
    if found == 0 || info.filename.is_null() {
        return Err(failure(LOAD, "could not locate the native component"));
    }
    let bytes = unsafe { CStr::from_ptr(info.filename) }.to_bytes();
    let module = Path::new(std::ffi::OsStr::from_bytes(bytes));
    // Never resolve an ambiguous loader path against the application's cwd.
    if !module.is_absolute() {
        return Err(failure(
            LOAD,
            "load the native library using an absolute path, or use dg_load",
        ));
    }
    module
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| failure(LOAD, "native component has no parent directory"))
}

#[cfg(target_os = "windows")]
#[cfg_attr(dg_embedded, allow(dead_code))]
fn component_directory() -> Result<PathBuf> {
    use std::ffi::{c_void, OsString};
    use std::os::windows::ffi::OsStringExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetModuleHandleExW(flags: u32, name: *const u16, module: *mut *mut c_void) -> i32;
        fn GetModuleFileNameW(module: *mut c_void, filename: *mut u16, size: u32) -> u32;
    }
    let mut module = ptr::null_mut();
    // FROM_ADDRESS | UNCHANGED_REFCOUNT: identify this library without loading it.
    if unsafe { GetModuleHandleExW(0x6, dg_is_yes_p as *const () as *const u16, &mut module) } == 0
    {
        return Err(failure(LOAD, "could not locate the native component"));
    }
    let mut path = vec![0_u16; 32768];
    let length =
        unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) } as usize;
    if length == 0 || length >= path.len() {
        return Err(failure(LOAD, "could not read the native component path"));
    }
    let module = PathBuf::from(OsString::from_wide(&path[..length]));
    module
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| failure(LOAD, "native component has no parent directory"))
}

#[cfg(not(any(unix, target_os = "windows")))]
#[cfg_attr(dg_embedded, allow(dead_code))]
fn component_directory() -> Result<PathBuf> {
    Err(failure(
        INCOMPATIBLE,
        "automatic component discovery is unsupported on this platform; use dg_load",
    ))
}

/// Evaluate using the model bundle adjacent to this native library.
#[no_mangle]
pub unsafe extern "C" fn dg_is_yes_p(
    content: *const c_char,
    content_bytes: usize,
    question: *const c_char,
    question_bytes: usize,
    criteria: *const DgCriteria,
    out_p_yes: *mut f64,
) -> i32 {
    boundary(|| {
        if out_p_yes.is_null() {
            return Err(failure(INVALID, "null probability output"));
        }
        let content = text(content, content_bytes)?;
        let question = text(question, question_bytes)?;
        let criteria = if criteria.is_null() {
            None
        } else {
            let criteria = &*criteria;
            Some((
                text(criteria.yes, criteria.yes_bytes)?,
                text(criteria.no, criteria.no_bytes)?,
            ))
        };
        if content.trim().is_empty() || question.trim().is_empty() {
            return Err(failure(INVALID, "content and question must be nonempty"));
        }
        question_prompt(question, criteria)?;
        *out_p_yes = with_default_model(|model| model.evaluate(content, question, criteria))?;
        Ok(())
    })
}

/// Run one operation on the lazily loaded bundle beside this native library.
fn with_default_model<T>(operation: impl FnOnce(&GateSession) -> Result<T>) -> Result<T> {
    let mut model = DEFAULT_MODEL
        .lock()
        .map_err(|_| failure(INTERNAL, "default model lock poisoned"))?;
    // Assign only after success: failed loads may be retried next call.
    #[cfg(dg_embedded)]
    if model.is_none() {
        *model = Some(load_embedded()?);
    }
    #[cfg(not(dg_embedded))]
    if model.is_none() {
        let directory = component_directory()?;
        let directory = directory
            .to_str()
            .ok_or_else(|| failure(INVALID, "component path is not UTF-8"))?;
        *model = Some(load(directory)?);
    }
    operation(
        model
            .as_ref()
            .ok_or_else(|| failure(INTERNAL, "default model missing"))?,
    )
}

// Borrowed option strings for one call; validated before any inference runs.
unsafe fn option_texts<'a>(
    options: *const *const c_char,
    option_bytes: *const usize,
    option_count: usize,
) -> Result<Vec<&'a str>> {
    if option_count < 2 {
        return Err(failure(INVALID, "choose needs at least two options"));
    }
    if option_count > MAX_OPTIONS {
        return Err(failure(INVALID, "too many options"));
    }
    if options.is_null() || option_bytes.is_null() {
        return Err(failure(INVALID, "null options"));
    }
    let pointers = slice::from_raw_parts(options, option_count);
    let lengths = slice::from_raw_parts(option_bytes, option_count);
    let mut texts = Vec::with_capacity(option_count);
    for (pointer, &length) in pointers.iter().zip(lengths) {
        let option = text(*pointer, length)?;
        if option.trim().is_empty() {
            return Err(failure(INVALID, "options must be nonempty"));
        }
        texts.push(option);
    }
    Ok(texts)
}

unsafe fn criteria_texts<'a>(criteria: *const DgCriteria) -> Result<Option<(&'a str, &'a str)>> {
    if criteria.is_null() {
        return Ok(None);
    }
    let criteria = &*criteria;
    Ok(Some((
        text(criteria.yes, criteria.yes_bytes)?,
        text(criteria.no, criteria.no_bytes)?,
    )))
}

// Copies a ranked result into caller-owned arrays only after complete success.
unsafe fn write_ranking(ranking: &[(usize, f64)], out_index: *mut i32, out_p: *mut f64) {
    for (rank, &(index, probability)) in ranking.iter().enumerate() {
        *out_index.add(rank) = index as i32;
        *out_p.add(rank) = probability;
    }
}

/// Rank options for a question using the bundle adjacent to this native library.
/// out_index and out_p receive option_count entries in descending probability.
#[no_mangle]
pub unsafe extern "C" fn dg_choose_p(
    content: *const c_char,
    content_bytes: usize,
    question: *const c_char,
    question_bytes: usize,
    options: *const *const c_char,
    option_bytes: *const usize,
    option_count: usize,
    criteria: *const DgCriteria,
    out_index: *mut i32,
    out_p: *mut f64,
) -> i32 {
    boundary(|| {
        if out_index.is_null() || out_p.is_null() {
            return Err(failure(INVALID, "null ranking output"));
        }
        let content = text(content, content_bytes)?;
        let question = text(question, question_bytes)?;
        let options = option_texts(options, option_bytes, option_count)?;
        let criteria = criteria_texts(criteria)?;
        let ranking =
            with_default_model(|model| model.choose(content, question, &options, criteria))?;
        write_ranking(&ranking, out_index, out_p);
        Ok(())
    })
}

/// Index of the most probable option, or -1 when its probability is below threshold.
#[no_mangle]
pub unsafe extern "C" fn dg_choose(
    content: *const c_char,
    content_bytes: usize,
    question: *const c_char,
    question_bytes: usize,
    options: *const *const c_char,
    option_bytes: *const usize,
    option_count: usize,
    criteria: *const DgCriteria,
    threshold: f64,
    out_index: *mut i32,
) -> i32 {
    boundary(|| {
        if out_index.is_null() {
            return Err(failure(INVALID, "null choice output"));
        }
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(failure(
                INVALID,
                "threshold must be finite and between zero and one",
            ));
        }
        let content = text(content, content_bytes)?;
        let question = text(question, question_bytes)?;
        let options = option_texts(options, option_bytes, option_count)?;
        let criteria = criteria_texts(criteria)?;
        let ranking =
            with_default_model(|model| model.choose(content, question, &options, criteria))?;
        let (index, probability) = ranking[0];
        *out_index = if probability >= threshold { index as i32 } else { -1 };
        Ok(())
    })
}

/// Boolean decision with a caller-selected, inclusive probability threshold.
#[no_mangle]
pub unsafe extern "C" fn dg_is_yes_at_threshold(
    content: *const c_char,
    content_bytes: usize,
    question: *const c_char,
    question_bytes: usize,
    criteria: *const DgCriteria,
    threshold: f64,
    out_yes: *mut u8,
) -> i32 {
    boundary(|| {
        if out_yes.is_null() {
            return Err(failure(INVALID, "null decision output"));
        }
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(failure(
                INVALID,
                "threshold must be finite and between zero and one",
            ));
        }
        let mut probability = 0.0;
        let status = dg_is_yes_p(
            content,
            content_bytes,
            question,
            question_bytes,
            criteria,
            &mut probability,
        );
        if status != 0 {
            return Err((status, LAST_ERROR.with(|error| error.borrow().clone())));
        }
        *out_yes = u8::from(probability >= threshold);
        Ok(())
    })
}

/// Boolean decision using a probability threshold of 0.5.
#[no_mangle]
pub unsafe extern "C" fn dg_is_yes(
    content: *const c_char,
    content_bytes: usize,
    question: *const c_char,
    question_bytes: usize,
    criteria: *const DgCriteria,
    out_yes: *mut u8,
) -> i32 {
    dg_is_yes_at_threshold(
        content,
        content_bytes,
        question,
        question_bytes,
        criteria,
        0.5,
        out_yes,
    )
}
thread_local! { static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) }; }

#[derive(Deserialize)]
struct Manifest {
    format_version: u32,
    model_id: String,
    max_tokens: usize,
    template_version: u32,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_choice_template")]
    choice_template_version: u32,
    sha256: HashMap<String, String>,
}

fn default_temperature() -> f64 {
    1.0
}

fn default_choice_template() -> u32 {
    1
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Template {
    QuestionFirst,
    ContentFirst,
}

impl Template {
    fn from_version(version: u32) -> Result<Self> {
        match version {
            1 => Ok(Self::QuestionFirst),
            2 => Ok(Self::ContentFirst),
            _ => Err(failure(INCOMPATIBLE, "unsupported template version")),
        }
    }

    fn pair<'a>(self, content: &'a str, prompt: &'a str) -> (&'a str, &'a str) {
        match self {
            Self::QuestionFirst => (prompt, content),
            Self::ContentFirst => (content, prompt),
        }
    }
}

fn question_prompt(question: &str, criteria: Option<(&str, &str)>) -> Result<String> {
    match criteria {
        None => Ok(question.to_string()),
        Some((yes, no)) => {
            if yes.trim().is_empty() || no.trim().is_empty() {
                return Err(failure(
                    INVALID,
                    "criteria descriptions must both be nonempty",
                ));
            }
            Ok(format!("{question}\nYes: {yes}\nNo: {no}"))
        }
    }
}

#[repr(C)]
pub struct DgCriteria {
    pub yes: *const c_char,
    pub yes_bytes: usize,
    pub no: *const c_char,
    pub no_bytes: usize,
}

pub struct GateSession {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_tokens: usize,
    template: Template,
    temperature: f64,
    choice_template: u32,
    metadata: String,
}

fn failure(code: i32, message: impl ToString) -> Failure {
    (code, message.to_string())
}
fn boundary(operation: impl FnOnce() -> Result<()>) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or_else(|_| Err(failure(INTERNAL, "internal panic")));
    match result {
        Ok(()) => {
            LAST_ERROR.with(|e| e.borrow_mut().clear());
            0
        }
        Err((code, message)) => {
            LAST_ERROR.with(|e| *e.borrow_mut() = message);
            code
        }
    }
}

// Non-null pointers must reference caller-owned readable memory of the stated size.
unsafe fn text<'a>(data: *const c_char, bytes: usize) -> Result<&'a str> {
    if bytes > MAX_INPUT_BYTES {
        return Err(failure(TOO_LONG, "input exceeds byte limit"));
    }
    if data.is_null() {
        return if bytes == 0 {
            Ok("")
        } else {
            Err(failure(INVALID, "null input pointer"))
        };
    }
    str::from_utf8(slice::from_raw_parts(data.cast::<u8>(), bytes))
        .map_err(|_| failure(INVALID, "input is not UTF-8"))
}

fn digest(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).map_err(|e| failure(LOAD, e))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let count = file.read(&mut buffer).map_err(|e| failure(LOAD, e))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn load(directory: &str) -> Result<GateSession> {
    if directory.is_empty() || directory.contains('\0') {
        return Err(failure(INVALID, "invalid bundle path"));
    }
    let directory = Path::new(directory)
        .canonicalize()
        .map_err(|e| failure(LOAD, e))?;
    let metadata =
        fs::read_to_string(directory.join("manifest.json")).map_err(|e| failure(LOAD, e))?;
    let manifest = parse_manifest(&metadata)?;
    for filename in ["model.onnx", "tokenizer.json"] {
        let expected = manifest
            .sha256
            .get(filename)
            .ok_or_else(|| failure(INCOMPATIBLE, format!("missing hash: {filename}")))?;
        if &digest(&directory.join(filename))? != expected {
            return Err(failure(INCOMPATIBLE, format!("hash mismatch: {filename}")));
        }
    }
    let tokenizer =
        Tokenizer::from_file(directory.join("tokenizer.json")).map_err(|e| failure(LOAD, e))?;
    start_runtime(&directory, &manifest)?;
    let session = session_builder()?
        .commit_from_file(directory.join("model.onnx"))
        .map_err(|e| failure(LOAD, e))?;
    finish(metadata, manifest, tokenizer, session)
}

// The standalone build (code/standalone) compiles this file with the bundle's
// manifest, tokenizer, and model inside the library and ONNX Runtime linked in.
// Its build script checks the embedded files against the manifest's hashes.
// The model goes in through the assembler's .incbin: include_bytes! on a
// 600 MB file makes the compiler run out of memory.
#[cfg(dg_embedded)]
std::arch::global_asm!(
    #[cfg(target_os = "windows")]
    ".section .rdata$dgmodel,\"dr\"",
    #[cfg(target_os = "macos")]
    ".section __TEXT,__const",
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    ".section .rodata.dgmodel,\"a\"",
    ".balign 64",
    #[cfg(target_os = "macos")]
    "_dg_model_start:",
    #[cfg(not(target_os = "macos"))]
    "dg_model_start:",
    concat!(".incbin \"", env!("DG_EMBED_DIR"), "/model.onnx\""),
    #[cfg(target_os = "macos")]
    "_dg_model_end:",
    #[cfg(not(target_os = "macos"))]
    "dg_model_end:",
    #[cfg(target_os = "macos")]
    ".private_extern _dg_model_start, _dg_model_end",
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    ".hidden dg_model_start, dg_model_end",
    #[cfg(not(target_os = "macos"))]
    ".globl dg_model_start, dg_model_end",
    #[cfg(target_os = "macos")]
    ".globl _dg_model_start, _dg_model_end",
);

#[cfg(dg_embedded)]
fn load_embedded() -> Result<GateSession> {
    extern "C" {
        static dg_model_start: u8;
        static dg_model_end: u8;
    }
    static MANIFEST: &str = include_str!(concat!(env!("DG_EMBED_DIR"), "/manifest.json"));
    static TOKENIZER: &[u8] = include_bytes!(concat!(env!("DG_EMBED_DIR"), "/tokenizer.json"));
    let model = unsafe {
        let start = ptr::addr_of!(dg_model_start);
        let end = ptr::addr_of!(dg_model_end);
        slice::from_raw_parts(start, end as usize - start as usize)
    };
    let manifest = parse_manifest(MANIFEST)?;
    let tokenizer = Tokenizer::from_bytes(TOKENIZER).map_err(|e| failure(LOAD, e))?;
    start_runtime(Path::new(""), &manifest)?;
    let session = session_builder()?
        .commit_from_memory(model)
        .map_err(|e| failure(LOAD, e))?;
    finish(MANIFEST.to_string(), manifest, tokenizer, session)
}

fn parse_manifest(metadata: &str) -> Result<Manifest> {
    let manifest: Manifest =
        serde_json::from_str(metadata).map_err(|e| failure(INCOMPATIBLE, e))?;
    Template::from_version(manifest.template_version)?;
    if manifest.choice_template_version != 1 {
        return Err(failure(INCOMPATIBLE, "unsupported choice template version"));
    }
    if manifest.format_version != 1
        || manifest.max_tokens == 0
        || manifest.max_tokens > 512
        || manifest.model_id.is_empty()
        || !manifest.temperature.is_finite()
        || !(0.01..=100.0).contains(&manifest.temperature)
    {
        return Err(failure(INCOMPATIBLE, "unsupported manifest"));
    }
    Ok(manifest)
}

/// Initialize ONNX Runtime once per process from the bundle's verified library.
#[cfg(not(dg_embedded))]
fn start_runtime(directory: &Path, manifest: &Manifest) -> Result<()> {
    let runtime_name = if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else {
        "libonnxruntime.so"
    };
    let runtime = directory
        .join(runtime_name)
        .canonicalize()
        .map_err(|e| failure(LOAD, e))?;
    let runtime_hash = digest(&runtime)?;
    let expected = manifest
        .sha256
        .get(runtime_name)
        .ok_or_else(|| failure(INCOMPATIBLE, "missing runtime hash"))?;
    if expected != &runtime_hash {
        return Err(failure(INCOMPATIBLE, "runtime hash mismatch"));
    }
    let runtime_path = runtime
        .to_str()
        .ok_or_else(|| failure(INVALID, "runtime path is not UTF-8"))?;
    initialize_runtime(runtime_hash, || {
        ort::init_from(runtime_path)
            .with_name("DecisionGator")
            .with_telemetry(false)
            .commit()
            .map(|_| ())
            .map_err(|e| failure(LOAD, e))
    })
}

/// The standalone library links ONNX Runtime in, so there is nothing to locate.
#[cfg(dg_embedded)]
fn start_runtime(_directory: &Path, _manifest: &Manifest) -> Result<()> {
    initialize_runtime("linked".to_string(), || {
        ort::init()
            .with_name("DecisionGator")
            .with_telemetry(false)
            .commit()
            .map(|_| ())
            .map_err(|e| failure(LOAD, e))
    })
}

fn initialize_runtime(identity: String, init: impl FnOnce() -> Result<()>) -> Result<()> {
    let mut initialized = RUNTIME
        .lock()
        .map_err(|_| failure(INTERNAL, "runtime lock poisoned"))?;
    if let Some(existing) = initialized.as_ref() {
        if existing != &identity {
            return Err(failure(
                INCOMPATIBLE,
                "a different ONNX Runtime is already loaded in this process",
            ));
        }
        return Ok(());
    }
    init()?;
    #[cfg(unix)]
    {
        let environment = ort::environment::get_environment().map_err(|e| failure(LOAD, e))?;
        EXIT_ENVIRONMENT.store(environment.ptr().cast_mut(), Ordering::SeqCst);
        if unsafe { atexit(release_environment_at_exit) } != 0 {
            return Err(failure(RESOURCE, "could not register runtime shutdown"));
        }
    }
    *initialized = Some(identity);
    Ok(())
}

fn session_builder() -> Result<ort::session::builder::SessionBuilder> {
    Session::builder()
        .map_err(|e| failure(LOAD, e))?
        .with_intra_threads(inference_threads())
        .map_err(|e| failure(RESOURCE, e))?
        .with_inter_threads(1)
        .map_err(|e| failure(RESOURCE, e))
}

fn finish(
    metadata: String,
    manifest: Manifest,
    mut tokenizer: Tokenizer,
    session: Session,
) -> Result<GateSession> {
    tokenizer
        .with_truncation(None)
        .map_err(|e| failure(INCOMPATIBLE, e))?;
    tokenizer.with_padding(None);
    let input_names: Vec<_> = session
        .inputs
        .iter()
        .map(|input| input.name.as_str())
        .collect();
    if input_names.len() != 3
        || !["input_ids", "attention_mask", "token_type_ids"]
            .iter()
            .all(|name| input_names.contains(name))
        || session.outputs.len() != 1
        || session.outputs[0].name != "logits"
    {
        return Err(failure(INCOMPATIBLE, "unexpected ONNX input/output names"));
    }
    Ok(GateSession {
        session: Mutex::new(session),
        tokenizer,
        max_tokens: manifest.max_tokens,
        template: Template::from_version(manifest.template_version)?,
        temperature: manifest.temperature,
        choice_template: manifest.choice_template_version,
        metadata,
    })
}

/// Choice template version 1: each option becomes a yes/no proposition.
fn choice_prompt(question: &str, option: &str, criteria: Option<(&str, &str)>) -> Result<String> {
    if question.trim().is_empty() || option.trim().is_empty() {
        return Err(failure(INVALID, "question and options must be nonempty"));
    }
    question_prompt(&format!("{question}\nAnswer: {option}"), criteria)
}

/// Stable descending ranking of softmax probabilities; ties keep option order.
fn rank(logits: &[f64]) -> Result<Vec<(usize, f64)>> {
    let peak = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits.iter().map(|z| (z - peak).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut ranking: Vec<(usize, f64)> = weights
        .iter()
        .enumerate()
        .map(|(index, weight)| (index, weight / total))
        .collect();
    if ranking
        .iter()
        .any(|(_, p)| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err(failure(INFERENCE, "invalid choice probability"));
    }
    ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(ranking)
}

impl GateSession {
    fn evaluate(
        &self,
        content: &str,
        question: &str,
        criteria: Option<(&str, &str)>,
    ) -> Result<f64> {
        if question.trim().is_empty() {
            return Err(failure(INVALID, "question is empty"));
        }
        let prompt = question_prompt(question, criteria)?;
        let probability = 1.0 / (1.0 + (-self.logit(content, &prompt)?).exp());
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(failure(INFERENCE, "invalid probability"));
        }
        Ok(probability)
    }

    /// Ranked (option index, probability) pairs; probabilities sum to one.
    fn choose(
        &self,
        content: &str,
        question: &str,
        options: &[&str],
        criteria: Option<(&str, &str)>,
    ) -> Result<Vec<(usize, f64)>> {
        if options.len() < 2 || options.len() > MAX_OPTIONS {
            return Err(failure(INVALID, "choose needs two to 256 options"));
        }
        debug_assert_eq!(self.choice_template, 1);
        let prompts = options
            .iter()
            .map(|option| choice_prompt(question, option, criteria))
            .collect::<Result<Vec<_>>>()?;
        let mut logits = Vec::with_capacity(options.len());
        for prompt in &prompts {
            logits.push(self.logit(content, prompt)?);
        }
        rank(&logits)
    }

    /// Calibrated log-odds of yes for one content/prompt pair.
    fn logit(&self, content: &str, prompt: &str) -> Result<f64> {
        if content.trim().is_empty() {
            return Err(failure(INVALID, "content is empty"));
        }
        let encoded = self
            .tokenizer
            .encode(self.template.pair(content, &prompt), true)
            .map_err(|e| failure(INVALID, e))?;
        let count = encoded.len();
        if count > self.max_tokens {
            return Err(failure(
                TOO_LONG,
                "input exceeds model token limit; input was not truncated",
            ));
        }
        let tensor = |values: &[u32]| {
            Tensor::from_array((
                [1_usize, count],
                values.iter().map(|&v| i64::from(v)).collect::<Vec<_>>(),
            ))
            .map_err(|e| failure(RESOURCE, e))
        };
        let ids = tensor(encoded.get_ids())?;
        let mask = tensor(encoded.get_attention_mask())?;
        let types = tensor(encoded.get_type_ids())?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| failure(INTERNAL, "model lock poisoned"))?;
        let output = session.run(ort::inputs!["input_ids" => ids, "attention_mask" => mask, "token_type_ids" => types]).map_err(|e| failure(INFERENCE, e))?;
        let (shape, logits) = output["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| failure(INFERENCE, e))?;
        if **shape != [1_i64, 2] || logits.len() != 2 || !logits.iter().all(|v| v.is_finite()) {
            return Err(failure(INFERENCE, "invalid model output"));
        }
        let logit = (f64::from(logits[1]) - f64::from(logits[0])) / self.temperature;
        if !logit.is_finite() {
            return Err(failure(INFERENCE, "invalid logit"));
        }
        Ok(logit)
    }
}

/// See decisiongator.h for the caller's pointer and lifetime obligations.
#[no_mangle]
pub unsafe extern "C" fn dg_load(
    path: *const c_char,
    path_bytes: usize,
    out_model: *mut *mut GateSession,
) -> i32 {
    boundary(|| {
        if out_model.is_null() {
            return Err(failure(INVALID, "null model output"));
        }
        *out_model = ptr::null_mut();
        let model = load(text(path, path_bytes)?)?;
        *out_model = Box::into_raw(Box::new(model));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn dg_evaluate(
    model: *mut GateSession,
    content: *const c_char,
    content_bytes: usize,
    question: *const c_char,
    question_bytes: usize,
    criteria: *const DgCriteria,
    out_p_yes: *mut f64,
) -> i32 {
    boundary(|| {
        if model.is_null() || out_p_yes.is_null() {
            return Err(failure(INVALID, "null model or probability output"));
        }
        let criteria = criteria_texts(criteria)?;
        let probability = (&*model).evaluate(
            text(content, content_bytes)?,
            text(question, question_bytes)?,
            criteria,
        )?;
        *out_p_yes = probability;
        Ok(())
    })
}

/// Explicit-session form of dg_choose_p with identical output rules.
#[no_mangle]
pub unsafe extern "C" fn dg_evaluate_choice(
    model: *mut GateSession,
    content: *const c_char,
    content_bytes: usize,
    question: *const c_char,
    question_bytes: usize,
    options: *const *const c_char,
    option_bytes: *const usize,
    option_count: usize,
    criteria: *const DgCriteria,
    out_index: *mut i32,
    out_p: *mut f64,
) -> i32 {
    boundary(|| {
        if model.is_null() || out_index.is_null() || out_p.is_null() {
            return Err(failure(INVALID, "null model or ranking output"));
        }
        let content = text(content, content_bytes)?;
        let question = text(question, question_bytes)?;
        let options = option_texts(options, option_bytes, option_count)?;
        let criteria = criteria_texts(criteria)?;
        let ranking = (&*model).choose(content, question, &options, criteria)?;
        write_ranking(&ranking, out_index, out_p);
        Ok(())
    })
}

unsafe fn copy_text(
    value: &str,
    buffer: *mut c_char,
    capacity: usize,
    required: *mut usize,
) -> Result<()> {
    if required.is_null() {
        return Err(failure(INVALID, "null required-size output"));
    }
    *required = value.len() + 1;
    if buffer.is_null() && capacity == 0 {
        return Ok(());
    }
    if buffer.is_null() {
        return Err(failure(INVALID, "null text output"));
    }
    if capacity < value.len() + 1 {
        return Err(failure(RESOURCE, "output buffer too small"));
    }
    ptr::copy_nonoverlapping(value.as_ptr(), buffer.cast::<u8>(), value.len());
    *buffer.add(value.len()) = 0;
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn dg_metadata(
    model: *const GateSession,
    buffer: *mut c_char,
    capacity: usize,
    required: *mut usize,
) -> i32 {
    boundary(|| {
        if model.is_null() {
            return Err(failure(INVALID, "null model"));
        }
        copy_text(&(*model).metadata, buffer, capacity, required)
    })
}

// Does not clear or overwrite the stored error, even on a size query.
#[no_mangle]
pub unsafe extern "C" fn dg_last_error(
    buffer: *mut c_char,
    capacity: usize,
    required: *mut usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        LAST_ERROR.with(
            |error| match copy_text(&error.borrow(), buffer, capacity, required) {
                Ok(()) => 0,
                Err((code, _)) => code,
            },
        )
    }))
    .unwrap_or(INTERNAL)
}

#[no_mangle]
pub unsafe extern "C" fn dg_release(model: *mut GateSession) {
    let _ = boundary(|| {
        if !model.is_null() {
            drop(Box::from_raw(model));
        }
        Ok(())
    });
}

/// Threads for one inference call. ONNX Runtime splits each operation evenly across its threads, so one
/// slow efficiency core holds all the others up: use only the fastest cores, one thread per physical core.
/// DECISIONGATOR_THREADS overrides this for applications that need cores for other work.
fn inference_threads() -> usize {
    if let Some(n) = std::env::var("DECISIONGATOR_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
    {
        return n.min(256);
    }
    let allowed = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    fast_cores().filter(|n| *n > 0).unwrap_or(allowed).min(allowed).clamp(1, 256)
}

#[cfg(target_vendor = "apple")]
fn fast_cores() -> Option<usize> {
    extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            old: *mut std::ffi::c_void,
            old_length: *mut usize,
            new: *mut std::ffi::c_void,
            new_length: usize,
        ) -> i32;
    }
    // perflevel0 is the performance cluster on Apple silicon; Intel Macs lack it and have one kind of core.
    for name in [c"hw.perflevel0.physicalcpu", c"hw.physicalcpu"] {
        let mut value: i32 = 0;
        let mut size = std::mem::size_of::<i32>();
        let status = unsafe {
            sysctlbyname(name.as_ptr(), (&mut value as *mut i32).cast(), &mut size, ptr::null_mut(), 0)
        };
        if status == 0 && value > 0 {
            return Some(value as usize);
        }
    }
    None
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn fast_cores() -> Option<usize> {
    let cpu_list = |text: &str| -> Option<Vec<usize>> {
        let mut cpus = Vec::new();
        for part in text.trim().split(',').filter(|p| !p.is_empty()) {
            let (a, b) = part.split_once('-').unwrap_or((part, part));
            cpus.extend(a.trim().parse::<usize>().ok()?..=b.trim().parse::<usize>().ok()?);
        }
        Some(cpus)
    };
    let number = |cpu: usize, file: &str| {
        fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/{file}"))
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
    };
    let online = cpu_list(&fs::read_to_string("/sys/devices/system/cpu/online").ok()?)?;
    // Intel hybrid chips list their performance cores here. ARM big.LITTLE chips rate each core's
    // capacity out of 1024; keep the big and middle clusters, drop the little one.
    let fast = if let Some(p_cores) = fs::read_to_string("/sys/devices/cpu_core/cpus").ok().and_then(|t| cpu_list(&t)) {
        online.iter().copied().filter(|c| p_cores.contains(c)).collect()
    } else if let Some(top) = online.iter().filter_map(|&c| number(c, "cpu_capacity")).max() {
        online.iter().copied().filter(|&c| number(c, "cpu_capacity").is_some_and(|v| v * 10 >= top * 6)).collect()
    } else {
        online
    };
    // Hyperthreads share a core's arithmetic units, so count physical cores.
    let cores: std::collections::HashSet<(u64, u64)> = fast
        .iter()
        .map(|&c| (number(c, "topology/physical_package_id").unwrap_or(0), number(c, "topology/core_id").unwrap_or(c as u64)))
        .collect();
    Some(cores.len())
}

#[cfg(windows)]
fn fast_cores() -> Option<usize> {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetLogicalProcessorInformationEx(relationship: i32, buffer: *mut u8, length: *mut u32) -> i32;
    }
    const RELATION_PROCESSOR_CORE: i32 = 0;
    let mut length = 0u32;
    unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, ptr::null_mut(), &mut length) };
    if length == 0 {
        return None;
    }
    let mut buffer = vec![0u8; length as usize];
    if unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buffer.as_mut_ptr(), &mut length) } == 0 {
        return None;
    }
    // One record per physical core: Relationship (u32), Size (u32), Flags (u8), EfficiencyClass (u8), ...
    // A higher efficiency class is a faster core; chips with one kind of core report 0 everywhere.
    let mut classes = Vec::new();
    let mut offset = 0usize;
    while offset + 10 <= (length as usize).min(buffer.len()) {
        let field = |at: usize| u32::from_le_bytes([buffer[at], buffer[at + 1], buffer[at + 2], buffer[at + 3]]);
        let size = field(offset + 4) as usize;
        if size == 0 {
            break;
        }
        if field(offset) == RELATION_PROCESSOR_CORE as u32 {
            classes.push(buffer[offset + 9]);
        }
        offset += size;
    }
    let top = *classes.iter().max()?;
    Some(classes.iter().filter(|&&c| c == top).count())
}

#[cfg(not(any(target_vendor = "apple", target_os = "linux", target_os = "android", windows)))]
fn fast_cores() -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inference_threads_uses_fast_cores_or_override() {
        let detected = inference_threads();
        assert!(detected >= 1);
        if let Some(fast) = fast_cores() {
            assert!(detected <= fast.max(1));
        }
        std::env::set_var("DECISIONGATOR_THREADS", "3");
        assert_eq!(inference_threads(), 3);
        std::env::set_var("DECISIONGATOR_THREADS", "zero");
        assert_eq!(inference_threads(), detected);
        std::env::remove_var("DECISIONGATOR_THREADS");
    }
    #[test]
    fn template_versions_preserve_pair_order_and_criteria() {
        let content = "The customer wants a booking.";
        let question = "Is this an appointment request?";
        for (criteria, expected_prompt) in [
            (None, question),
            (
                Some(("Requests a booking", "Other requests")),
                "Is this an appointment request?\nYes: Requests a booking\nNo: Other requests",
            ),
        ] {
            let prompt = question_prompt(question, criteria).unwrap();
            assert_eq!(prompt, expected_prompt);
            assert_eq!(
                Template::from_version(1).unwrap().pair(content, &prompt),
                (expected_prompt, content)
            );
            assert_eq!(
                Template::from_version(2).unwrap().pair(content, &prompt),
                (content, expected_prompt)
            );
        }
    }

    #[test]
    fn unsupported_templates_and_empty_criteria_are_rejected() {
        for version in [0, 3, u32::MAX] {
            assert_eq!(Template::from_version(version).unwrap_err().0, INCOMPATIBLE);
        }
        for criteria in [("", "No"), ("Yes", " \n"), ("", "")] {
            assert_eq!(
                question_prompt("Question?", Some(criteria)).unwrap_err().0,
                INVALID
            );
        }
    }

    #[test]
    fn invalid_abi_inputs_do_not_write_probability() {
        unsafe {
            let mut probability = 42.0;
            assert_eq!(
                dg_evaluate(
                    ptr::null_mut(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    0,
                    ptr::null(),
                    &mut probability
                ),
                INVALID
            );
            assert_eq!(probability, 42.0);
            assert_eq!(dg_load(ptr::null(), 0, ptr::null_mut()), INVALID);
            dg_release(ptr::null_mut());
        }
    }
    #[test]
    fn invalid_text_and_missing_bundle_are_rejected() {
        unsafe {
            assert!(text([0xff_u8].as_ptr().cast(), 1).is_err());
            let path = "a-directory-that-does-not-exist";
            let mut model = ptr::dangling_mut();
            assert_eq!(dg_load(path.as_ptr().cast(), path.len(), &mut model), LOAD);
            assert!(model.is_null());
        }
    }
    #[test]
    fn errors_can_be_copied_without_being_cleared() {
        assert_eq!(boundary(|| Err(failure(INVALID, "bad input"))), INVALID);
        unsafe {
            let mut required = 0;
            assert_eq!(dg_last_error(ptr::null_mut(), 0, &mut required), 0);
            assert_eq!(required, 10);
            let mut buffer = [42_i8; 16];
            assert_eq!(
                dg_last_error(buffer.as_mut_ptr(), 2, &mut required),
                RESOURCE
            );
            assert_eq!(buffer[0], 42);
            assert_eq!(
                dg_last_error(buffer.as_mut_ptr(), buffer.len(), &mut required),
                0
            );
            assert_eq!(
                str::from_utf8(slice::from_raw_parts(buffer.as_ptr().cast(), 9)).unwrap(),
                "bad input"
            );
            assert_eq!(buffer[9], 0);
        }
    }
    #[test]
    fn choice_prompts_and_ranking() {
        assert_eq!(
            choice_prompt("Which team?", "billing", None).unwrap(),
            "Which team?\nAnswer: billing"
        );
        assert_eq!(
            choice_prompt("Which team?", "billing", Some(("A", "B"))).unwrap(),
            "Which team?\nAnswer: billing\nYes: A\nNo: B"
        );
        assert_eq!(choice_prompt("Q?", " ", None).unwrap_err().0, INVALID);
        let ranking = rank(&[0.0, 2.0, 0.0]).unwrap();
        assert_eq!(ranking[0].0, 1);
        assert_eq!((ranking[1].0, ranking[2].0), (0, 2));
        let total: f64 = ranking.iter().map(|r| r.1).sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!(ranking[0].1 > 0.75);
        // Ties keep caller order; extreme logits stay finite.
        let tie = rank(&[1.0, 1.0]).unwrap();
        assert_eq!((tie[0].0, tie[1].0), (0, 1));
        let extreme = rank(&[1000.0, -1000.0]).unwrap();
        assert_eq!(extreme[0], (0, 1.0));
    }

    #[test]
    fn invalid_choice_abi_inputs_do_not_write_outputs() {
        unsafe {
            let mut index = [7_i32; 2];
            let mut p = [42.0_f64; 2];
            let content = "c";
            let question = "q";
            let one = ["a".as_ptr().cast::<c_char>()];
            let one_len = [1_usize];
            assert_eq!(
                dg_evaluate_choice(
                    ptr::null_mut(),
                    content.as_ptr().cast(),
                    1,
                    question.as_ptr().cast(),
                    1,
                    one.as_ptr(),
                    one_len.as_ptr(),
                    1,
                    ptr::null(),
                    index.as_mut_ptr(),
                    p.as_mut_ptr()
                ),
                INVALID
            );
            assert_eq!(option_texts(one.as_ptr(), one_len.as_ptr(), 1).unwrap_err().0, INVALID);
            assert_eq!(option_texts(ptr::null(), ptr::null(), 2).unwrap_err().0, INVALID);
            let blank = ["a".as_ptr().cast::<c_char>(), " ".as_ptr().cast::<c_char>()];
            let blank_len = [1_usize, 1];
            assert_eq!(option_texts(blank.as_ptr(), blank_len.as_ptr(), 2).unwrap_err().0, INVALID);
            let two = ["a".as_ptr().cast::<c_char>(), "b".as_ptr().cast::<c_char>()];
            assert_eq!(option_texts(two.as_ptr(), blank_len.as_ptr(), 2).unwrap(), ["a", "b"]);
            assert_eq!(
                dg_choose(
                    content.as_ptr().cast(),
                    1,
                    question.as_ptr().cast(),
                    1,
                    two.as_ptr(),
                    blank_len.as_ptr(),
                    2,
                    ptr::null(),
                    1.5,
                    index.as_mut_ptr()
                ),
                INVALID
            );
            assert_eq!(index, [7, 7]);
            assert_eq!(p, [42.0, 42.0]);
        }
    }

    #[test]
    fn panic_is_contained_at_boundary() {
        assert_eq!(boundary(|| panic!("test panic")), INTERNAL);
    }
}
