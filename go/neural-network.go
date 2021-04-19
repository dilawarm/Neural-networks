package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"./datasets"

	"gonum.org/v1/gonum/mat"
)

type Layer_Dense struct {
	weights *mat.Dense
	biases  *mat.Dense
	output  *mat.Dense
}

func NewLayer_Dense(n_inputs int, n_neurons int) *Layer_Dense {
	w := make([]float64, n_inputs*n_neurons)
	for i := range w {
		w[i] = -1 + rand.Float64()*(1-(-1))
	}
	return &Layer_Dense{weights: mat.NewDense(n_inputs, n_neurons, w), biases: mat.NewDense(1, n_neurons, nil)}
}

func (m *Layer_Dense) forward(inputs *mat.Dense) {
	var res mat.Dense
	res.Mul(inputs, m.weights)
	m.output = mat.NewDense(res.RawMatrix().Rows, res.RawMatrix().Cols, nil)

	for i := 0; i < res.RawMatrix().Rows; i++ {
		for j := 0; j < res.RawMatrix().Cols; j++ {
			m.output.Set(i, j, res.At(i, j)+m.biases.At(0, j))
		}
	}
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

type Activation_ReLU struct {
	output *mat.Dense
}

func (a *Activation_ReLU) forward(inputs *mat.Dense) {
	a.output = mat.NewDense(inputs.RawMatrix().Rows, inputs.RawMatrix().Cols, nil)

	for i := 0; i < inputs.RawMatrix().Rows; i++ {
		for j := 0; j < inputs.RawMatrix().Cols; j++ {
			a.output.Set(i, j, math.Max(0, inputs.At(i, j)))
		}
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())
	X, _ := datasets.Spiral_data(100, 3)

	layer1 := NewLayer_Dense(2, 5)
	activation1 := Activation_ReLU{}

	layer1.forward(X)
	matPrint(layer1.output)
	activation1.forward(layer1.output)
	matPrint(activation1.output)
}