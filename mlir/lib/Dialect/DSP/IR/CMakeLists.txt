add_mlir_dialect_library(MLIRDSPDialect
  DSPDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/DSP

  DEPENDS
  
  # MLIRDSPEnumsGen
  # MLIRDSPAttributesIncGen
  # MLIRDSPIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRSideEffectInterfaces
  # DSPShapeInferenceInterfaceIncGen
  )
