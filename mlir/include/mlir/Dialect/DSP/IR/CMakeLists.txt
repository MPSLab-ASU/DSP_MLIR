add_mlir_dialect(DSP dsp)
add_mlir_doc(DSP DSP Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DSP.td)
mlir_tablegen(DSPEnums.h.inc -gen-enum-decls)
mlir_tablegen(DSPEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRDSPEnumsGen)

set(LLVM_TARGET_DEFINITIONS DSP.td)
mlir_tablegen(DSPAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=dsp)
mlir_tablegen(DSPAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=dsp)
add_public_tablegen_target(MLIRDSPAttributesIncGen)

