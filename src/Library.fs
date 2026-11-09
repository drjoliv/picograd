// picograd: a tiny neural-network library in the style of Andrej Karpathy's micrograd.
//
// Big picture for learners:
//   1. A neural network is just a big math expression built from + and * (and a squashing function).
//   2. We record that expression as a graph of Node.T values, so every number "remembers" how it was made.
//   3. Training = (a) run the network forward to get a prediction, (b) measure how wrong it is (the loss),
//      (c) walk the graph backwards to learn how much each weight contributed to the error (gradients,
//      a.k.a. backpropagation), (d) nudge each weight a little in the direction that lowers the loss.
//   4. Repeat many times. The loss goes down, and the weights end up encoding what was learned.
//
// File layout, from low level to high level:
//   Feature : describes how to turn a raw data row (e.g. a CSV row) into numbers.
//   Node    : one number in the computation graph + automatic differentiation (backpropagation).
//   Dot     : draws a computation graph with Graphviz (debugging/visualisation only).
//   Neuron  : one unit that computes tanh(w1*x1 + w2*x2 + ... + bias).
//   Layer   : a list of neurons that all read the same inputs.
//   MLP     : "multi-layer perceptron", a chain of layers; includes the training loop.
namespace picograd

open System
open System.Globalization


/// Feature: a named rule for extracting ONE number from a raw data item (for example, the "Age" of a CSV row).
/// [AutoOpen] means that after `open picograd` you can use `feature` and `Feature.map` without a prefix.
[<AutoOpen>]
module Feature =

    /// A named rule that turns one raw data row into one number the network can read.
    /// The type parameter 'T is whatever a raw row looks like (a CSV row type, a record, ...).
    type Feature<'T> = {
        /// Human readable name of the feature, for example "Age".
        Name: string
        /// Function that takes a raw row and returns the number the network will see.
        Value: 'T -> float
    }

    /// Small helper to build a Feature without writing the record syntax.
    let feature name value =
        { Name = name; Value = value }

    /// Converts raw rows into the two things training needs:
    /// the first result has one list of numbers per row (the model INPUTS, one number per feature),
    /// the second result has one number per row (the ANSWER we want the model to learn, for example
    /// +1 for survived and -1 for died).
    let map (features: Feature<'T> list) (target: Feature<'T>) (inputs: 'T list) : float list list * float list =
        let len = List.length features
        let go feature input = feature.Value input

        // For every row, apply every feature, giving a row of numbers like [age; class; sex; ...].
        let items =
            inputs
            |> List.map (fun input ->  
                features
                |> List.map (fun feature -> go feature input)
            )

        // The target feature is applied to each row to get the "right answer" for that row.
        let targets = inputs |> List.map target.Value

        items, targets

/// Node: the heart of the library. Each Node.T is a single number in a computation graph.
/// It stores its value, which operation created it, which nodes it was computed from, and (after
/// backpropagation) its gradient. This is called "automatic differentiation".
module Node =
    open FSharp.Data
    open System.Net.Http

    /// The kind of operation that produced a node. We need to remember it so that backpropagation
    /// knows which derivative rule to apply.
    type Operation =
        /// A leaf: an input value or a weight, not computed from anything.
        | None
        /// a + b
        | Addition
        /// a * b
        | Multiplication
        /// a raised to a fixed power n, where n is stored in the case.
        | Exponentiation of float
        /// tanh(a): squashes any number into the range between -1 and 1.
        | HyperbolicTangent
        /// e raised to the power a.
        | Exp


        /// Human readable name, used when drawing the graph.
        member this.Display =
            match this with
            | Operation.None -> "None"
            | Operation.Addition -> "Addition"
            | Operation.Multiplication -> "Multiplication"
            | Operation.HyperbolicTangent -> "HyperbolicTangent"
            | Operation.Exponentiation(_) -> "Exponentiation"
            | Operation.Exp -> "Exp"            

    /// A node in the computation graph.
    /// Data is the actual number (the "forward" value). It is mutable because training updates weights in place.
    /// Operation is how this number was produced.
    /// Children are the nodes it was computed FROM (its inputs). Empty for leaves.
    /// Name is an optional label, only used for drawing graphs.
    /// Gradient is d(loss)/d(this node), that is: "if I nudge this number up a tiny bit, how much does the
    /// final loss change?". It is filled in by backpropagation. Positive means "increasing this makes the
    /// loss worse", so training moves the value the opposite way.
    [<StructuredFormatDisplay("Node(Value = {Data}; Name= {Name}; Operation={Operation})")>]
    type T = {
        /// The number this node holds.
        mutable Data : float
        /// The operation that produced this node.
        Operation : Operation
        /// The nodes this node was computed from.
        Children : T list
        /// Optional label for graph drawings.
        Name: string
        /// How much the final loss changes per unit change of Data. Set by backpropagation.
        mutable Gradient: float
    } with

        // Operator overloads. They let us write ordinary maths such as `a * b + c` on Nodes, while
        // secretly building the graph: each operator creates a NEW node whose Children are its operands.

        /// node + node
        static member (+) (a: T, b: T) =
            { Data = a.Data + b.Data; Operation = Operation.Addition; Children = [a; b]; Name = String.Empty; Gradient = 0.0 }

        /// node + plain number: the number is wrapped in a leaf node first, then added as above.
        static member (+) (a: T, b: float) =
            if Double.IsNaN b then invalidArg (nameof b) "Node value cannot be NaN"
            a + { Data = b; Operation = Operation.None; Children = []; Name = String.Empty; Gradient = 0.0 }

        /// plain number + node
        static member (+) (a: float, b: T) =
            if Double.IsNaN a then invalidArg (nameof a) "Node value cannot be NaN"
            { Data = a; Operation = Operation.None; Children = []; Name = String.Empty; Gradient = 0.0 } + b

        /// node * node
        static member (*) (a: T, b: T) =
            { Data = a.Data * b.Data ; Operation = Operation.Multiplication;  Children = [a; b];  Name = String.Empty; Gradient = 0.0 }

        /// node * plain number
        static member (*) (a: T, b: float) =
            if Double.IsNaN b then invalidArg (nameof b) "Node value cannot be NaN"
            a * { Data = b; Operation = Operation.None; Children = []; Name = String.Empty; Gradient = 0.0 }

        /// plain number * node
        static member (*) (a: float, b: T) =
            if Double.IsNaN a then invalidArg (nameof a) "Node value cannot be NaN"
            { Data = a; Operation = Operation.None; Children = []; Name = String.Empty; Gradient = 0.0 } * b

        /// Negation: -a is built as a * -1, so no new operation type is needed.
        static member (~-) (a: T) =
            a * (-1.0)

        /// Subtraction is built from addition and negation: a - b = a + (-b).
        static member ( - ) (a: T, b: T) =
            a + (-b)

        /// node - plain number
        static member ( - ) (a: T, b: float) =
            if Double.IsNaN b then invalidArg (nameof b) "Node value cannot be NaN"
            a + (-b)

        /// plain number - node
        static member ( - ) (a: float, b: T) =
            if Double.IsNaN a then invalidArg (nameof a) "Node value cannot be NaN"
            a + (-b)

        /// Raise a node to a fixed numeric power. Defining a member named Pow is what makes the
        /// F# ** operator work on this type, for example `x ** 2.0`.
        static member Pow (a: T, exponent: float) =
            if Double.IsNaN a.Data then invalidArg (nameof a) "Node value cannot be NaN"
            { Data = Math.Pow(a.Data, exponent); Operation = Operation.Exponentiation exponent; Children = [a]; Name = String.Empty; Gradient = 0.0 }

        /// Division is built from multiplication and a power: a / b = a * b^(-1).
        static member (/) (a: T, b: T) =
            a * b ** -1.0


    /// Creates a leaf node holding a plain number (an input feature or a weight).
    /// NaN ("not a number") is rejected immediately, so bad data fails loudly at the source instead of
    /// silently corrupting every later calculation.
    let value (data : float) = 
        if Double.IsNaN data then invalidArg (nameof data) "Node value cannot be NaN"
        { Data = data; Operation = Operation.None; Children = []; Name = String.Empty; Gradient = 0.0 }

    /// Same as `value`, but with a name (useful when drawing the graph).
    let create (data : float) (name: string) = 
        if Double.IsNaN data then invalidArg (nameof data) "Node value cannot be NaN"
        { Data = data; Operation = Operation.None; Children = []; Name = name; Gradient = 0.0 }

    /// Returns a copy of the node with a new name. It is a COPY, so its identity differs from the original.
    let label (name: string) (node : T) = 
        { node with Name = name }

    /// Multiplies the two nodes in the tuple and names the result.
    let multiply (name: string) (a: T, b: T) =
        a * b |> label name

    /// Divides the first node of the tuple by the second and names the result.
    let divide (name: string) (a: T, b: T) =
        a / b |> label name

    /// Adds the two nodes in the tuple and names the result.
    let add (name: string) (a: T, b: T) =
        a + b |> label name

    /// e raised to the power of a.
    let exp (name: string) (a: T) =
        {
            Data = Math.Exp a.Data
            Operation = Operation.Exp
            Children = [a]
            Name = name
            Gradient = 0.0
        }

    /// tanh is the "activation function". Without it, stacking layers would collapse into one big linear
    /// formula and the network could only learn straight-line relationships. tanh bends the output into
    /// a smooth S-curve between -1 and 1, which lets the network learn non-linear patterns.
    /// Warning: for very large positive or negative inputs the output saturates near -1 or 1 and its slope
    /// (the gradient) becomes almost 0, so learning stalls. That is why input features are scaled to small ranges.
    let tanh (name: string) (a: T) =
        let data = Math.Tanh a.Data
        { Data = data; Operation = Operation.HyperbolicTangent; Children = [a]; Name = name; Gradient = 0.0 }

    /// Marks a node as the final output of the computation: d(loss)/d(loss) = 1.
    /// Backpropagation starts from here and works backwards.
    let root (node: T) : T =
        node.Gradient <- 1.0
        node

    // F# has no built-in `+=` for mutable record fields, so define one that accumulates into Gradient.
    // It ADDS instead of assigning because a node can be used in several places in the graph, and the
    // gradient is the sum of the contributions from every place it is used (the multivariable chain rule).
    let inline private (+=) (node: T) (delta: float) : unit =
        node.Gradient <- node.Gradient + delta

    // Parents-before-children order: every node appears before all of its children,
    // so a node's gradient is complete by the time it is propagated.
    // (A "parent" is a node computed from others; its "children" are the inputs it was computed from.)
    // Implementation: a depth-first walk that records a node AFTER visiting its children, which gives
    // children-first order; reversing it gives parents-first order.
    let private topologicalOrder (root: T) : T list =
        // Reference equality: two different nodes with the same numbers must not be treated as one.
        // The visited set also ensures a node shared by several parents is only processed once.
        let visited = System.Collections.Generic.HashSet<T>(HashIdentity.Reference)
        let order = System.Collections.Generic.List<T>()

        let rec visit (node: T) =
            if visited.Add node then
                for child in node.Children do
                    visit child
                order.Add node

        visit root
        order |> Seq.rev |> List.ofSeq

    // The LOCAL derivative rule for each operation. Given this node's gradient (how much the loss cares
    // about this node), push a share of it down to each child using the chain rule:
    //   child.Gradient += (d node / d child) * node.Gradient
    let private propagateLocal (node: T) : unit =
        match node.Operation with
        | Operation.None -> ()   // leaf: nothing below it to update
        | Operation.Addition ->
            // d(a+b)/da = 1 and d(a+b)/db = 1, so both children receive the gradient unchanged.
            node.Children.[0] += node.Gradient
            node.Children.[1] += node.Gradient
        | Operation.Multiplication ->
            // d(a*b)/da = b and d(a*b)/db = a: each child's gradient is scaled by the OTHER child's value.
            let left = node.Children.[0]
            let right = node.Children.[1]
            left += right.Data * node.Gradient
            right += left.Data * node.Gradient
        | Operation.HyperbolicTangent ->
            // d tanh(x)/dx = 1 - tanh(x)^2. node.Data already holds tanh(x), so we reuse it.
            node.Children.[0] += (1.0 - (node.Data * node.Data)) * node.Gradient
        | Operation.Exp ->
            // The derivative of e^x is e^x itself, which is node.Data.
            node.Children.[0] += node.Data * node.Gradient
        | Operation.Exponentiation exponent ->
            // Power rule: d(x^n)/dx = n * x^(n-1).
            let child = node.Children.[0]
            child += exponent * (child.Data ** (exponent - 1.0)) * node.Gradient

    /// Backpropagation: process every node from the output back to the inputs, once each, in an order
    /// where all of a node's parents have already contributed to its gradient.
    /// Afterwards every node's Gradient says how the loss would change if that node's value changed.
    let backward (node: T) : unit =
        topologicalOrder node |> List.iter propagateLocal

    /// Convenience: set the output's gradient to 1, then run backward.
    let propagate = root >> backward


/// Dot: renders a computation graph as a picture using Graphviz. This is for understanding and
/// debugging only and plays no part in training.
module Dot =
    open FSharp.Data
    open System.Diagnostics
    open System.IO

    /// Produces the text of a Graphviz digraph describing the graph below `root`.
    /// Each value node becomes a box showing name, data and gradient; each operation becomes its own
    /// small node (+, *, tanh, ...) with arrows from its inputs to it and from it to its result.
    let string (root: Node.T) =
        let nodes = ResizeArray<string>()
        let edges = ResizeArray<string>()
        let mutable nextId = 0

        // Walks the graph, emitting nodes/edges, and returns the id of the node it just drew.
        let rec visit (node: Node.T) =
            let id = sprintf "n%d" nextId
            nextId <- nextId + 1

            // Debug guard: stop if a node is exactly zero (usually a sign of a dead/broken value).
            if node.Data = 0.0 then
                printfn "Node data is zero: %A" node.Children
                failwith "Node data is zero"

            let label = sprintf "    %s [label=\"{%s|data: %.4f| %.4f}\"; shape=\"record\"];" id node.Name node.Data node.Gradient
            nodes.Add label

            if node.Operation <> Node.Operation.None then
                // This node was computed by an operation, so draw an extra node for the operator itself.
                let opid = sprintf "%s%O" id node.Operation.Display
                let op_label =
                    match node.Operation with
                    | Node.Operation.None                            -> failwith ""
                    | Node.Operation.Addition                        -> sprintf "    %s [label=\"%s\"];" opid "+"
                    | Node.Operation.Multiplication                  -> sprintf "    %s [label=\"%s\"];" opid "*"
                    | Node.Operation.HyperbolicTangent               -> sprintf "    %s [label=\"%s\"];" opid "tanh"
                    | Node.Operation.Exp                             -> sprintf "    %s [label=\"%s\"];" opid "exp"
                    | Node.Operation.Exponentiation exponent  -> sprintf "    %s [label=\"%s%.4f\"];" opid "**" exponent

                nodes.Add op_label
                edges.Add(sprintf "    %s -> %s;" opid id)

                // Arrows go from each input, into the operator.
                for child in node.Children do
                    let childid = visit child
                    edges.Add(sprintf "    %s -> %s;" childid opid)

                id

            else
                // Leaf nodes have no children, so this branch normally does nothing.
                for child in node.Children do
                    let childid = visit child
                    edges.Add(sprintf "    %s -> %s;" childid id)

                id


        visit root |> ignore

        let lines = seq {
            yield "digraph Computation {"
            yield "    rankdir=LR;"   // lay the graph out left to right
            yield! nodes
            yield! edges
            yield "}"
        }

        String.Join(Environment.NewLine, lines)

    /// Sends the Graphviz text to the kroki.io web service (needs internet), which returns an SVG image.
    /// The image is saved to graph.svg in the current directory and its path is returned.
    let generate (graph: string) =
        let response =
            Http.Request(
                "https://kroki.io/graphviz/svg",
                httpMethod = HttpMethod.Post,
                headers = [ HttpRequestHeaders.ContentType "text/plain" ],
                body = HttpRequestBody.TextRequest graph
            )

        let path = System.IO.Path.GetFullPath("graph.svg")
        match response.Body with
        | Text svg ->
            System.IO.File.WriteAllText(path, svg)
        | Binary binary ->
            let svg = System.Text.Encoding.UTF8.GetString binary
            System.IO.File.WriteAllText(path, svg)

        path


    // Opens the file with the default program for its type (for an SVG, usually a web browser).
    let private access (path: string) =
        let fullPath = Path.GetFullPath(path)
    
        Process.Start(
            ProcessStartInfo(
                fullPath,
                UseShellExecute = true
            )
        )
        |> ignore

    /// Full pipeline: graph, Graphviz text, SVG file, then open it. Usage: `Dot.view someNode`.
    let view = string >> generate >> access

/// Neuron: the basic unit of a neural network, loosely inspired by a brain cell.
/// It takes several inputs, multiplies each by its own learned weight, adds them up with a bias,
/// and squashes the result with tanh:   output = tanh(w1*x1 + w2*x2 + ... + bias)
module Neuron =
    open System

    // Random number generator used to pick the starting weights. Seeded (42) so that every run starts
    // from the same weights and gives the same result, which is essential for comparing experiments.
    let mutable private rng = Random(42)

    /// Resets the random generator. Call it before creating a network to reproduce the same initial weights.
    let setSeed (seed: int) = rng <- Random(seed)

    /// A random number uniformly distributed between min and max.
    let uniform (min, max) = min + (max - min) * rng.NextDouble()

    /// A list of `len` random numbers between -1 and 1.
    let uniforms len = 
        [ for i in 1 .. len -> uniform(-1.0, 1.0) ]

    /// One neuron.
    type T = {
        /// One weight per input; they control how much each input matters. These are what training learns.
        Weights: Node.T list
        /// A learned constant added after the weighted sum; it lets the neuron shift its output up or
        /// down independent of the inputs.
        Bias: Node.T
    } with
        /// Every learnable number in this neuron. The training loop adjusts exactly these.
        member this.Parameters = this.Weights @ [this.Bias]

        /// Forward pass when the inputs are already graph nodes. Keeping them as nodes keeps the graph
        /// connected, so gradients can flow back through it.
        static member Call(neuron: T, inputs: Node.T list) : Node.T =
            let sum =
                List.zip neuron.Weights inputs                           // pair each weight with its input
                |> List.map (fun x -> Node.multiply String.Empty x)      // w_i * x_i
                |> List.reduce (+)                                       // add them all together
            sum + neuron.Bias |> Node.tanh String.Empty                  // add bias, then squash with tanh

        /// Forward pass for plain numbers: each one is wrapped in a leaf node first.
        static member Call(neuron: T, inputs: float list) : Node.T =
            T.Call(neuron, inputs |> List.map Node.value)

    /// Creates a neuron that accepts `len` inputs, with random starting weights and bias in the range -1 to 1.
    /// Starting from RANDOM values (not all zeros) is important: if every weight were identical, all
    /// neurons would compute and learn the same thing.
    let create (len : int) : T =
        let weights = [ for i in 1 .. len -> Node.value <| uniform(-1.0, 1.0) ]
        printf "Creating neuron with %d weights\n" (List.length weights)

        let bias = Node.value <| uniform(-1.0, 1.0) |> Node.label "bias"
        { Weights = weights; Bias = bias }

    /// Runs the neuron on one input row. The inputs may be a Node.T list or a float list; the matching
    /// Call overload on T is resolved at compile time.
    let inline call (neuron: T) (inputs: ^I) : Node.T =
        ((^I or T): (static member Call: T * ^I -> Node.T) (neuron, inputs))

    //let forward (neuron: T) (inputs: Node.T list) : Node.T = call neuron inputs
    // Partially apply to get a plain function: let f = Neuron.create 3 |> Neuron.toFunction
    //let toFunction (neuron: T) : Node.T list -> Node.T = forward neuron

/// Layer: several neurons side by side. Every neuron in the layer receives the SAME inputs but has its
/// own weights, so each one can learn to detect a different pattern.
module Layer = 
    /// A layer of neurons.
    type T = {
        /// The neurons in this layer.
        Neurons: Neuron.T list
    } with
        /// All learnable numbers in the layer.
        member this.Parameters = this.Neurons |> List.collect (fun n -> n.Parameters)

        /// Forward pass: feed the inputs to every neuron. The result has one output per neuron.
        static member Call(layer: T, inputs: Node.T list) : Node.T list =
            layer.Neurons |> List.map (fun neuron -> Neuron.call neuron inputs)

        /// Forward pass for plain numbers: each one is wrapped in a leaf node first.
        static member Call(layer: T, inputs: float list) : Node.T list=
            T.Call(layer, inputs |> List.map Node.value)

    /// Creates a layer. `numin` is how many inputs each neuron takes (the size of the previous layer, or
    /// the number of features for the first layer). `nout` is how many neurons, and therefore how many
    /// outputs, the layer has.
    let create (numin: int) (nout: int) : T =
        let neurons = [ for i in 1 .. nout -> Neuron.create numin ]
        { Neurons = neurons }

    /// Runs the layer on one input row. Accepts a Node.T list or a float list.
    let inline call (layer: T) (inputs: ^I list) : Node.T list =
        ((^I or T): (static member Call: T * ^I list -> Node.T list) (layer, inputs))

/// MLP: multi-layer perceptron, the classic "feed-forward neural network". The outputs of one layer
/// become the inputs of the next. Example: nin = 6 with layers [4; 1] means 6 input features,
/// then a layer of 4 neurons (each with 6 weights), then a layer of 1 neuron (with 4 weights), which
/// gives the final answer.
module MLP =
    /// A network made of layers applied in order.
    type T = {
        /// The layers, from the one that reads the features to the output layer.
        Layers: Layer.T list
    } with

        /// Every learnable number in the whole network (all weights and biases).
        member this.Parameters = this.Layers |> List.collect (fun l -> l.Parameters)

        // Reset all gradients to 0. This must happen before each backward pass, because gradients
        // accumulate (+=) and would otherwise include leftovers from the previous iteration.
        member private this.Zero() : unit = 
            for parameter in this.Parameters do
                parameter.Gradient <- 0.0

        // One gradient-descent step: move every parameter a small distance AGAINST its gradient.
        // 0.1 is the learning rate. Too small = learns very slowly. Too large = overshoots and can diverge
        // or saturate the tanh units.
        member private this.Step() : unit = 
            for parameter in this.Parameters do
                parameter.Data <- parameter.Data - 0.1 * parameter.Gradient

        // Builds an untrained network. `size` lists how many numbers flow between layers, e.g. [6; 4; 1];
        // layer i maps size[i] inputs to size[i+1] outputs.
        static member private Create(nin: int, layers: int list) : T =
            let size = [nin] @ layers
            let layers =
                [ for i in 0 .. layers.Length - 1 ->
                    Layer.create size.[i] size.[i+1]]
            { Layers = layers }

        /// The training loop (full-batch gradient descent). Each iteration:
        /// 1. Forward pass: predict an output for every training row.
        /// 2. Loss: mean squared error, the average of (prediction - target) squared. 0 means perfect.
        ///    Squaring makes every error positive and punishes big mistakes more than small ones.
        /// 3. Zero the old gradients.
        /// 4. Backpropagate: compute d(loss)/d(parameter) for every weight and bias.
        /// 5. Step: nudge each parameter in the direction that reduces the loss.
        /// The number of passes over the data ("epochs") is fixed at 2000.
        member this.Learn(inputs: float list list, targets: float list) : unit =
            for i in 1 .. 2000 do
                let predictions = [for input in inputs ->  T.Call (this, input) ] |> List.concat
                let loss =
                    let squaredErrors = [for (y, p) in List.zip targets predictions -> (p - y)**2]
                    (squaredErrors |> List.reduce (+)) * (1.0 / float squaredErrors.Length)   // sum / count = mean
                this.Zero()
                Node.propagate loss
                this.Step()

                // Print occasionally so you can watch the loss fall. If it doesn't fall, something is wrong.
                if i % 200 = 0 then
                    printfn "Iteration %d, Loss: %A" i loss.Data
            ()

        /// Convenience trainer that starts from raw data rows (see the Feature module): converts the rows
        /// to numbers, shuffles them with a fixed seed so the split is reproducible, splits them, creates a
        /// network with a fixed seed, trains it and returns it.
        /// NOTE: `size` below is 20% of the rows, but `training` takes the FIRST `size` rows, so this trains
        /// on only 20% of the data and the remaining 80% (`test`) is unused. The usual convention is the
        /// reverse: train on 80% and hold out 20% for validation.
        static member Fit<'T>(features: Feature<'T> list, target: Feature<'T>, items: 'T list, nin: int, layers: int list) : T =

            let (inputs, targets) = Feature.map features target items

            let shuffled =
                let rng = Random(1234)
                List.zip inputs targets |> List.sortBy (fun _ -> rng.Next())

            let split = 0.20
            let size =  float (shuffled |> List.length) * split |> int
            let test = shuffled |> List.take size
            let training = shuffled |> List.skip size

            let trainX, trainY = List.unzip training
            Neuron.setSeed 42
            let mlp = T.Create(nin, layers)
            mlp.Learn(trainX, trainY)
            mlp
            

        
        /// Placeholder overload for already-numeric data; not implemented yet.
        static member Fit(inputs: float list list, targets: float list, nin: int, layers: int list) : T =
            raise <| System.NotImplementedException()
//            let mlp = MLP.create nin layers
            //mlp.Learn(inputs, targets)
            //mlp

        /// Forward pass through the whole network: feed the inputs through each layer in order, using
        /// each layer's outputs as the next layer's inputs. The result is the network's final output(s).
        static member Call(mlp: T, inputs: Node.T list) : Node.T list =
            let mutable inputs = inputs
            for layer in mlp.Layers do
                inputs <- Layer.call layer inputs
            inputs

        /// Forward pass for plain numbers: each one is wrapped in a leaf node first.
        static member Call(mlp: T, inputs: float list) : Node.T list=
            T.Call(mlp, inputs |> List.map Node.value)

    /// Builds an untrained network: `nin` input features, then one layer per entry in `nouts`
    /// (each entry is the number of neurons in that layer). Example: MLP.create 6 [4; 1].
    let create nin nouts =
        let size = [nin] @ nouts
        let layers =
            [ for i in 0 .. nouts.Length - 1 ->
                Layer.create size.[i] size.[i+1]]
        { Layers = layers }

    /// Runs the network on one input row. Accepts a float list or a Node.T list (resolved at compile time).
    /// Example: (MLP.call net [0.4; 1.0; ...]).Head.Data is the network's prediction as a plain number.
    let inline call (mlp: T) (inputs: ^I list) : Node.T list =
        ((^I or T): (static member Call: T * ^I list -> Node.T list) (mlp, inputs))