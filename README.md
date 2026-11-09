# picograd

picograd is a small F# neural-network and automatic-differentiation library inspired by
[micrograd](https://github.com/karpathy/micrograd). It builds computation graphs from scalar values,
propagates gradients through those graphs, and uses them to train simple multilayer perceptrons.

## Features

- Scalar computation nodes with automatic differentiation and backpropagation.
- Neurons, layers, and multilayer perceptrons (MLPs) with `tanh` activation.
- Helpers for mapping records or other data rows into numeric features and targets.
- Graphviz DOT output for inspecting computation graphs.

## Requirements

- .NET SDK 10.0 or later.

## Getting started

Create a graph with `Node` values and ordinary arithmetic operators, then call `Node.propagate` to
calculate gradients:

```fsharp
open picograd

let x = Node.value 2.0
let weight = Node.value -3.0
let loss = x * weight

Node.propagate loss

printfn "Output: %f" loss.Data
printfn "d(loss)/d(x): %f" x.Gradient
printfn "d(loss)/d(weight): %f" weight.Gradient
```

Build a small network and run a forward pass:

```fsharp
open picograd

Neuron.setSeed 42
let model = MLP.create 2 [4; 1]
let outputs = MLP.call model [0.2; 0.8]
let prediction = outputs.Head.Data

printfn "Prediction: %f" prediction
```

`MLP.create 2 [4; 1]` creates a network with two input features, one hidden layer of four neurons,
and one output neuron. `MLP.call` returns graph nodes, so gradients can also be propagated through
the result.

To train a model, provide input rows and matching targets to `Learn`:

```fsharp
Neuron.setSeed 42
let model = MLP.create 2 [4; 1]

let inputs = [
    [0.0; 0.0]
    [0.0; 1.0]
    [1.0; 0.0]
    [1.0; 1.0]
]
let targets = [-1.0; 1.0; 1.0; -1.0]

model.Learn(inputs, targets)
```

Training currently uses full-batch gradient descent for 2,000 iterations, with a fixed learning
rate. The example uses targets in the range `[-1, 1]`, matching the network's `tanh` output.

## Inspecting computation graphs

`Dot.string` returns Graphviz DOT text for a node and its ancestors:

```fsharp
let graph = Dot.string loss
printfn "%s" graph
```

`Dot.view` also sends the DOT text to the Kroki service to create and open an SVG. Do not use it
with graphs containing sensitive data; use `Dot.string` when you only need local DOT output.

## Install from GitHub Packages

Packages are published to the GitHub NuGet feed for this repository on pushes to `main`. The CI
workflow assigns each published package a version of the form `1.0.0-ci.<run number>`.

Configure the feed with a GitHub token that has `read:packages` access:

```sh
dotnet nuget add source "https://nuget.pkg.github.com/drjoliv/index.json" --name github --username YOUR_GITHUB_USERNAME --password YOUR_GITHUB_TOKEN --store-password-in-clear-text
```

Then install a specific published version:

```sh
dotnet add package picograd --version 1.0.0-ci.RUN_NUMBER --source github
```

Replace `RUN_NUMBER` with the run number from the GitHub Actions workflow that published the
package. Keep your token private. See GitHub's documentation for
[authenticating to GitHub Packages](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-nuget-registry).

## Build and test from source

```sh
dotnet restore
dotnet build --configuration Release --no-restore
dotnet run --configuration Release --no-build
```

The last command runs the Expecto test executable.
