package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	inputs := mat.NewDense(4, 1, []float64{1, 2, 3, 2.5})
	weights := mat.NewDense(3, 4, []float64{0.2, 0.8, -0.5, 1.0, 0.5, -0.91, 0.26, -0.5, -0.26, -0.27, 0.17, 0.87})
	biases := mat.NewDense(3, 1, []float64{2, 3, 0.5})
	output := mat.NewDense(3, 1, nil)
	output.Product(weights, inputs)
	output.Add(output, biases)
	fmt.Println(output.RawMatrix().Data)
}
