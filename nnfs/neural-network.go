package main

import (
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Layer_Dense struct {
	weights *mat.Dense
	biases  *mat.Dense
	output  *mat.Dense
}

func NewLayer_Dense(n_inputs int, n_neurons int) Layer_Dense {
	rand.Seed(time.Now().UnixNano())
	w := make([]float64, n_inputs*n_neurons)
	for i := range w {
		w[i] = -0.1 + rand.Float64()*(0.1-(-0.1))
	}
	weights := mat.NewDense(n_inputs, n_neurons, w)
	return Layer_Dense{weights: weights, biases: nil, output: nil}
}

func (m *Layer_Dense) forward(inputs *mat.Dense) {
	inputRows, _ := inputs.Dims()
	_, weightCols := m.weights.Dims()

	m.output = mat.NewDense(inputRows, weightCols, nil)
	m.output.Product(inputs, m.weights)

	m.biases = mat.NewDense(inputRows, weightCols, nil)
	m.output.Add(m.output, m.biases)
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func main() {
	X := mat.NewDense(3, 4, []float64{1, 2, 3, 2.5, 2.0, 5.0, -1.0, 2.0, -1.5, 2.7, 3.3, -0.8})

	layer1 := NewLayer_Dense(4, 5)
	layer2 := NewLayer_Dense(5, 2)

	layer1.forward(X)
	matPrint(layer1.output)
	layer2.forward(layer1.output)
	matPrint(layer2.output)
}
