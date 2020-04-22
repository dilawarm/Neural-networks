package main

import "fmt"

func main() {
	inputs := []float32{1.2, 5.1, 2.1}
	weights := []float32{3.1, 2.1, 8.7}
	bias := 3

	output := inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + float32(bias)
	fmt.Println(output)
}
