namespace Picograd.Test

module NodeTest =

    open Expecto
    open picograd

    let private finiteFloat value =
        if System.Double.IsFinite value then value % 100.0 else 0.0

    [<Tests>]
    let nodeTests =
        testList "Node" [
            testProperty "addition sums the data values" <| 
                fun (a: float) (b: float)->
                    let a = finiteFloat a
                    let b = finiteFloat b
                    let left = Node.value a
                    let right = Node.value b

                    Expect.isTrue ((left + right).Data.Equals(a + b)) "The result should contain the sum of both data values"

            testProperty "multiplication multiplies the data values" <|
                fun (a: float) (b: float) ->
                    let a = finiteFloat a
                    let b = finiteFloat b
                    let left = Node.value a
                    let right = Node.value b

                    Expect.isTrue ((left * right).Data.Equals(a * b)) "The result should contain the product of both data values"

            testProperty "addition followed by multiplication combines both operations" <|
                fun (a: float) (b: float) (c: float) ->
                    let a = finiteFloat a
                    let b = finiteFloat b
                    let c = finiteFloat c
                    let left = Node.value a
                    let middle = Node.value b
                    let right = Node.value c

                    Expect.isTrue (((left + middle) * right).Data.Equals((a + b) * c)) "The result should add the first two values, then multiply by the third"

            testCase "DOT output contains the arithmetic computation graph" <| fun _ ->
                let expression = (Node.value 2.0 + Node.value 3.0) * Node.value 4.0
                let graph = Dot.string expression

                Expect.isTrue (graph.Contains("digraph Computation {")) "DOT should contain the graph header"
                Expect.isTrue (graph.Contains("n2 -> n1Addition;")) "DOT should connect the addition operands"
                Expect.isTrue (graph.Contains("n1 -> n0Multiplication;")) "DOT should connect the addition to the multiplication"

            testCase "propagation runs on the computation graph" <| fun _ ->
                let x1 = Node.create 2 "x1"
                let x2 = Node.create 0 "x2"

                // weights
                let w1 = Node.create -3 "w1"
                let w2 = Node.create 1 "w2"

                // bias of the neuron
                let b = Node.create 6.8813735870195432 "b"

                let x1w1 = Node.multiply  "x1w1" (x1, w1)
                let x2w2 = Node.multiply  "x2w2" (x2, w2)
                let x1w1x2w2 = Node.add "x1w1 + x2w2" (x1w1, x2w2)
                let n = Node.add "n" (x1w1x2w2, b) 

                //let o = Node.tanh "o" n
                let e = (2.0 * n) |> Node.exp "e"
                let o = (e - 1.0) / (e  + 1.0)
                Node.propagate o
        ]

    [<EntryPoint>]
    let main args =
        runTestsWithCLIArgs [] args nodeTests
