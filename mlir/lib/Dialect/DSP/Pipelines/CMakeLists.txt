add_mlir_dialect_library(MLIRDSPPipelines
  DSPPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/DSP

  LINK_LIBS PUBLIC
  MLIRDSPDialect
  #MLIRDSPTransforms
  MLIRMemRefTransforms
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
)
