add_mlir_conversion_library(MLIRDSPToAffine
  DSPToAffine.cpp
  DSPToAffinePass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/DSP
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRAffineDialect
  MLIRIR
  MLIRPass
  MLIRDSPDialect
  MLIRDSPTransforms
  MLIRSupport
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRTransformUtils
  MLIRTensorDialect
  )
