package main

import (
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func main() {
	rand.Seed(time.Now().UnixNano())
	w := make([]float64, 4*3)
	for i := range w {
		w[i] = -0.1 + rand.Float64()*(0.1-(-0.1))
	}
	weights := mat.NewDense(4, 3, w)
	matPrint(weights)
}
