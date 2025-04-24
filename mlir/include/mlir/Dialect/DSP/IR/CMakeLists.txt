add_mlir_dialect(DSP dsp)
add_mlir_doc(DSP DSP Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DSP.td)
#mlir_tablegen(DSP_Ops.h.inc -gen-op-decls)
#mlir_tablegen(DSP_Ops.cpp.inc -gen-op-defs)
#mlir_tablegen(DSP_Dialect.h.inc -gen-dialect-decls)
#mlir_tablegen(DSP_Dialect.cpp.inc -gen-dialect-defs)
#add_public_tablegen_target(MLIRDspOpsIncGen)
mlir_tablegen(DSPEnums.h.inc -gen-enum-decls)
mlir_tablegen(DSPEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRDSPEnumsGen)





# set(LLVM_TARGET_DEFINITIONS DSP.td)
# mlir_tablegen(DSPAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=dsp)
# mlir_tablegen(DSPAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=dsp)
# add_public_tablegen_target(MLIRDSPAttributesIncGen)

# set(LLVM_TARGET_DEFINITIONS DSP.td)
# mlir_tablegen(DSPAttrDefs.h.inc -gen-attrdef-decls 
#               -attrdefs-dialect=dsp)
# mlir_tablegen(DSPAttrDefs.cpp.inc -gen-attrdef-defs 
#               -attrdefs-dialect=dsp)
# add_public_tablegen_target(DSPAttrDefsIncGen)
