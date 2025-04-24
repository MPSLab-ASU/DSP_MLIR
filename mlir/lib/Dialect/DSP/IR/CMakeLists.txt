set(LLVM_OPTIONAL_SOURCES
  DSPDialect.cpp
  ToyCombine.cpp
  )

set(LLVM_TARGET_DEFINITIONS ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(MLIRToyCombineIncGen)

add_mlir_dialect_library(MLIRDSPDialect
  DSPDialect.cpp
  ToyCombine.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/DSP

  DEPENDS
  MLIRShapeInferenceOpInterfacesIncGen
  
  #DspSimpleBlocksOpsIncGen
  #DspSimpleBlocksCombineIncGen

  # MLIRDSPEnumsGen
  # MLIRDSPAttributesIncGen
  # MLIRDSPIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRShapedOpInterfaces
  
  #MLIRToyCombineIncGen
  )
