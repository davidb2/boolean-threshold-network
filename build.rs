fn main() -> Result<(), Box<dyn std::error::Error>> {
  // Tell Cargo to re-run this script if any .proto changes:
  println!("cargo:rerun-if-changed=protos/message.proto");
  // println!("cargo:rerun-if-changed=build.rs");

  // Tell prost-build where to find your .proto files
  let proto_files = &["protos/message.proto"];
  let proto_dirs  = &["protos"];
  prost_build::compile_protos(proto_files, proto_dirs)?;
  Ok(())
}
