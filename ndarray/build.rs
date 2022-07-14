extern crate bindgen;

use std::env;
use std::error::Error;
use std::fs::read_to_string;
use std::path::PathBuf;
use std::io::Write;
use std::collections::{HashMap, HashSet};

use bindgen::callbacks::ParseCallbacks;


/* ----------------------------- files compiled with nvcc ----------------------------- */

const CUDA_OPS: &[&'static str] = &[
    "add", 
    "mul", 
    "sub", 
    "div"
];

/* ------------------------------------------------------------------------------------ */

#[derive(Debug)]
struct AddFromPrimitive {
    namemap: HashMap<String, String>,
    nameset: HashSet<String>,
}

impl AddFromPrimitive {
    fn new(namemap_tuples: Vec<(String, String)>) -> AddFromPrimitive {Self { 
        namemap: HashMap::from_iter(namemap_tuples.clone()),
        nameset: HashSet::from_iter(namemap_tuples.iter().map(|x| x.1.clone())),
    }}
    fn boxed(self) -> Box<AddFromPrimitive> {
        Box::new(self)
    }
}

impl ParseCallbacks for AddFromPrimitive {
    fn add_derives(&self, _name: &str) -> Vec<String> {
        if self.nameset.contains(_name) {vec![
            "num_enum::FromPrimitive".to_owned()
        ]} else {vec![]}
    }
    fn item_name(&self, _original_item_name: &str) -> Option<String> {
        match self.namemap.get(_original_item_name) {
            Some(v) => Some(v.clone()),
            None => None
        }
    }
}

fn insert_before(path: PathBuf, pat : String, insert_str: String) {
    let content = read_to_string(path.clone()).unwrap();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(path).unwrap();
    write!(file, "{}", content.replace(&pat, &{insert_str + &pat})).unwrap();
}

fn cuda_build(
    cuda_include_name: &[&str], /* the cuda header files to include */
    cuda_library_name: &[&str], /* the cuda libraries to link */
    cuda_include_dir: &str,     /* the directory where cuda.h lives */
    cuda_library_dir: &str,     /* the directory where *.lib file or lib*.a file lives */
    target_dir: PathBuf,         /* the target directory where the bindings goes */
) -> Result<(), Box<dyn Error>> {
    let mut header_content = String::new();

    for x in cuda_include_name {
        header_content += &format!("#include \"{cuda_include_dir}/{x}.h\"");
        header_content += "\n";
    }

    let header_path = target_dir.join("header.h");
    let mut header_file = std::fs::File::create(&header_path)?;
    header_file.write_all(header_content.as_bytes())?;

    /* set library path */
    println!("cargo:rustc-link-search={}", cuda_library_dir);
    /* generate lib bindings */
    for x in cuda_library_name {
        println!("cargo:rustc-link-lib={x}");
        let out_path = target_dir.join(x.to_string()+".rs");
        {let bindings = bindgen::Builder::default()
            /* here are some options */
            .parse_callbacks(AddFromPrimitive::new(vec![("cudaError_enum".to_string(), "CudaError".to_string())]).boxed())
            .rustified_enum("cudaError_enum")
            .no_copy("CUstream_st")
            .header(header_path.as_path().to_str().unwrap().to_owned())
            .generate()
            ?;
        bindings
            .write_to_file(out_path.clone())?;}
        // Some string manipulation tricks
        insert_before(out_path, "CUDA_ERROR_UNKNOWN".to_string(), "#[default]\n    ".to_string());
    }
    Ok(())
}

fn ptx_build (
    source_dir: PathBuf,
    target_dir: PathBuf,
    cu_file_name_list: &[&str],
) -> Result<(), Box<dyn Error>> {
    for cu_file_name in cu_file_name_list {
        std::process::Command::new("nvcc")
            .arg(source_dir.join(cu_file_name.to_string() + ".cu"))
            .arg("-o").arg(target_dir.join(cu_file_name.to_string() + ".ptx"))
            .arg("--ptx")
            .spawn()?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {

    let ptx_out_dir = PathBuf::from(env::var("OUT_DIR")? + "/ptx");
    let cu_file_dir = PathBuf::from(env::current_dir()?).join("cusrc");
    if std::fs::create_dir(&ptx_out_dir).is_ok() {
        ptx_build(
            cu_file_dir, 
            ptx_out_dir,
            CUDA_OPS
        )?;
    }

    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_INCLUDE_PATH");

    let cuda_out_dir = PathBuf::from(env::var("OUT_DIR")? + "/nvidia_toolkit");
    if std::fs::create_dir(&cuda_out_dir).is_ok() {
        cuda_build(
            &["cuda", "cuda_runtime", "cuda_runtime_api"],
            &["cuda", "cudart", "cudadevrt", "cudart_static"],
            &env::var("CUDA_INCLUDE_PATH")?,
            &env::var("CUDA_LIBRARY_PATH")?,
            cuda_out_dir,
        )?;
    }

    Ok(())
}
