namespace Picograd.Test

module NodeTest =

    open Expecto
    open picograd

    [<Tests>]
    let nodeTests =
        testList "Node" [
            testProperty "addition sums the data values" <| 
                fun (a: float) (b: float)->
                    let left = Node.value a
                    let right = Node.value b

                    Expect.isTrue ((left + right).Data.Equals(a + b)) "The result should contain the sum of both data values"

            testProperty "multiplication multiplies the data values" <|
                fun (a: float) (b: float) ->
                    let left = Node.value a
                    let right = Node.value b

                    Expect.isTrue ((left * right).Data.Equals(a * b)) "The result should contain the product of both data values"

            testProperty "addition followed by multiplication combines both operations" <|
                fun (a: float) (b: float) (c: float) ->
                    let left = Node.value a
                    let middle = Node.value b
                    let right = Node.value c

                    Expect.isTrue (((left + middle) * right).Data.Equals((a + b) * c)) "The result should add the first two values, then multiply by the third"

            testCase "DOT output contains the computation graph" <| fun _ ->
                let expression = (Node.value 2.0 + Node.value 3.0) * Node.value 4.0
                let newline = System.Environment.NewLine
                let expected = System.String.Join(newline, [|
                    "digraph Node {"
                    "    node [shape=record];"
                    "    rankdir=LR;"
                    "    n0 [label=\"*\\n20\"];"
                    "    n1 [label=\"+\\n5\"];"
                    "    n2 [label=\"2\"];"
                    "    n3 [label=\"3\"];"
                    "    n4 [label=\"4\"];"
                    "    n2 -> n1;"
                    "    n3 -> n1;"
                    "    n1 -> n0;"
                    "    n4 -> n0;"
                    "}"
                |])

                Expect.equal (Dot.string expression) expected "DOT should represent each node and its operand edges"

            testCase "DOT output contains the computation graph" <| fun _ ->
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
