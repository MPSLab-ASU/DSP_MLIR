//===- dspc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/AST.h"
#include "toy/Dialect.h"
#include "toy/DebugConfig.h"
#include "toy/Lexer.h"
#include "toy/MLIRGen.h"
#include "toy/Parser.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

//Added by ABhinav
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

// using namespace mlir;
using namespace dsp;

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input dsp file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Toy, MLIR };
} // namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Toy, "dsp", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action {
  None,
  DumpAST,
  DumpMLIR,
  DumpMLIRAffine,
  DumpMLIRLinalg,
  DumpMLIRLLVM,
  DumpLLVMIR,
  DumpLLVMIRHexagonV68,
  RunJIT
};
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                          "output the MLIR dump after affine lowering")),
    cl::values(clEnumValN(DumpMLIRLinalg, "mlir-linalg",
                          "output the MLIR dump after tosa lowering")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
	cl::values(clEnumValN(DumpLLVMIRHexagonV68, "llvm-hexagonv68", "output the LLVM IR dump where target is hexagonv68")),
    cl::values(
        clEnumValN(RunJIT, "jit",
                   "JIT the code and run it by invoking the main function")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));
static cl::opt<bool> affineIn("affineIn", cl::desc("Input is affine file"));
static cl::opt<bool> enableAffineOpt("affineOpt", cl::desc("Enable optimizations"));
static cl::opt<bool> enableCanonicalOpt("canonOpt", cl::desc("Enable optimizations"));

// Add this line to expose the variable to other files
bool getEnableCanonicalOpt() {
    return enableCanonicalOpt;
}

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<dsp::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Handle '.dsp' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).ends_with(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    module = mlirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  DEBUG_PRINT_NO_ARGS() ;
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp> &module) {
  if (int error = loadMLIR(context, module))
    return error;

  mlir::PassManager pm(module.get()->getName());
  // DEBUG_PRINT_NO_ARGS() ;
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // Check to see what granularity of MLIR we are compiling to.
  bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
  bool isLoweringTosaToLinalg = emitAction >= Action::DumpMLIRLinalg;
  bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;
  // bool isAffineToLLVM = emitAction >= Action::DumpMLIRAffineToLLVM;
  // DEBUG_PRINT_NO_ARGS() ;
  if (enableOpt || isLoweringToAffine ) {
    // Inline all functions into main and then delete them.
    DEBUG_PRINT_WITH_ARGS("Going for inliner pass \n");
    pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    mlir::OpPassManager &optPM = pm.nest<mlir::dsp::FuncOp>();
    DEBUG_PRINT_WITH_ARGS("Going for shape inference pass \n");

    optPM.addPass(mlir::dsp::createShapeInferencePass());
    // optPM.addPass(mlir::createCanonicalizerPass());
    // optPM.addPass(mlir::createCSEPass());
  }

  if (isLoweringToAffine) {
    // Partially lower the dsp dialect.
    pm.addPass(mlir::dsp::createLowerToAffinePass());

    // Add a few cleanups post lowering.
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    // optPM.addPass(mlir::createCanonicalizerPass());
    // optPM.addPass(mlir::createCSEPass());

    // Add optimizations if enabled.
    if (enableOpt) {
    // if (0) {
      DEBUG_PRINT_WITH_ARGS("Opt enabled --loop fusion &  \n");
      DEBUG_PRINT_WITH_ARGS("Opt enabled --createAffineScalarReplacementPass \n");
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());
      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());

      //all affine optimizations
      // llvm::errs() << "All affine optimizations\n" ;
      // // optPM.addPass(mlir::affine::createAffineDataCopyGenerationPass());
      // optPM.addPass(mlir::affine::createLoopFusionPass());
      // optPM.addPass(mlir::affine::createAffineLoopInvariantCodeMotionPass());
      // // optPM.addPass(mlir::affine::createLoopTilingPass());
      // optPM.addPass(mlir::affine::createLoopUnrollPass());
      // optPM.addPass(mlir::affine::createLoopUnrollAndJamPass());
      // optPM.addPass(mlir::affine::createPipelineDataTransferPass());
      // optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
      // // optPM.addPass(mlir::affine::createAffineVectorizePass()); //no-option
      // // optPM.addPass(mlir::affine::createAffineParallelizePass());
      // optPM.addPass(mlir::affine::createAffineLoopNormalizePass());
      // optPM.addPass(mlir::affine::createLoopCoalescingPass());
      // optPM.addPass(mlir::affine::createSimplifyAffineStructuresPass());
      // optPM.addPass(mlir::affine::createAffineExpandIndexOpsPass());

      //All mlir-optimizations
      optPM.addPass(mlir::createCSEPass());
    }

  }
    mlir::OpPassManager &optPM2 = pm.nest<mlir::func::FuncOp>();
    // Add optimizations if enabled.
    if(enableCanonicalOpt) {
      DEBUG_PRINT_WITH_ARGS("Canonical Pass Enabled \n");
      optPM2.addPass(mlir::createCanonicalizerPass());
      optPM2.addPass(mlir::createCSEPass());
      // optPM2.addPass(mlir::affine::createLoopFusionPass());
      // optPM2.addPass(mlir::affine::createAffineScalarReplacementPass());
      // optPM2.addPass(mlir::createCSEPass());
    }

    if (enableAffineOpt) {
      DEBUG_PRINT_WITH_ARGS("Opt enabled --loop fusion &  \n");
      DEBUG_PRINT_WITH_ARGS("Opt enabled --createAffineScalarReplacementPass \n");
      optPM2.addPass(mlir::affine::createLoopFusionPass());
      optPM2.addPass(mlir::affine::createAffineScalarReplacementPass());
      optPM2.addPass(mlir::createCSEPass());
    }

    if(enableCanonicalOpt) {
      DEBUG_PRINT_WITH_ARGS("Canonical Pass Enabled \n");
      optPM2.addPass(mlir::createCanonicalizerPass());
      optPM2.addPass(mlir::createCSEPass());
      // optPM2.addPass(mlir::affine::createLoopFusionPass());
      // optPM2.addPass(mlir::affine::createAffineScalarReplacementPass());
      // optPM2.addPass(mlir::createCSEPass());
    }

  if (isLoweringTosaToLinalg) {
    // Partially lower the dsp dialect.
    // pm.addPass(mlir::dsp::createLowerTosaToLinalgPass());

    // Add a few cleanups post lowering.
    // mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaOptionalDecompositions());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaInferShapesPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaMakeBroadcastablePass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::tosa::createTosaToLinalgNamed());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    // TODO: Remove pass that operates on const tensor and enable optionality
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaLayerwiseConstantFoldPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaMakeBroadcastablePass());
    // optPM.addPass(mlir::tosa::createTosaValidation());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::dsp::createLowerTosaToLinalgPass());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

  }

  if (isLoweringToLLVM) {
    // Finish lowering the dsp IR to the LLVM dialect.
    pm.addPass(mlir::dsp::createLowerToLLVMPass());
    // This is necessary to have line tables emitted and basic
    // debugger working. In the future we will add proper debug information
    // emission directly from our frontend.
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());
  // enableOpt = false;

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

int dumpLLVMIRHexagonV68(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  LLVMInitializeHexagonTarget();
  LLVMInitializeHexagonTargetInfo();
  LLVMInitializeHexagonTargetMC();
  LLVMInitializeHexagonAsmPrinter();

  // Configure the LLVM Module
  /*
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }

  
  llvm::HexagonTargetMachine &TM = getHexagonTargetMachine();
  
  
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
														TM.get().get());
                                                        //tmOrError.get().get());
														
  //Failed to make Hexagon Target Machine
  */
														
  // enableOpt = false;

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}


int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

int main(int argc, char **argv) {
  //Get time of execution of code
  auto start = std::chrono::high_resolution_clock::now();

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "dsp compiler\n");

  if (emitAction == Action::DumpAST)
    return dumpAST();

  // If we aren't dumping the AST, then we are compiling with/to MLIR.
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);

  mlir::MLIRContext context(registry);
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::dsp::DspDialect>();
  //Added by abhinav to support affine and arith input files
  //Todo -- put a check for which option
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect, mlir::affine::AffineDialect, mlir::tosa::TosaDialect, 
                    mlir::linalg::LinalgDialect, mlir::BuiltinDialect, mlir::math::MathDialect, mlir::LLVM::LLVMDialect,
                    mlir::scf::SCFDialect>();
  context.appendDialectRegistry(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module))
    return error;

  // If we aren't exporting to non-mlir, then we are done.
  bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
  if (isOutputingMLIR) {
    module->dump();
    return 0;
  }

  // Check to see if we are compiling to LLVM IR.
  if (emitAction == Action::DumpLLVMIR)
    return dumpLLVMIR(*module);

  else if (emitAction == Action::DumpLLVMIRHexagonV68)
    return dumpLLVMIRHexagonV68(*module);

  // Otherwise, we must be running the jit.
  if (emitAction == Action::RunJIT)
  {
    // auto start = std::chrono::high_resolution_clock::now();
    int getJitRunStatus = runJit(*module);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double , milli> duration = (end - start);
    // std::chrono::duration<double> duration = (end - start); //this will give in secs
    
    
    // std::cout << "\nTime taken: " << duration.count() << " millisecs" << "\n";
    return getJitRunStatus;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double , milli> duration = (end - start);  
  std::cout << "\nTime taken: " << duration.count() << " millisecs" << "\n";
  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return -1;
}
