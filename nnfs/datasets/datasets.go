package datasets

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func Spiral_data(points int, classes int) (*mat.Dense, *mat.Dense) {
	X := mat.NewDense(points*classes, 2, nil)
	y := mat.NewDense(points*classes, 1, nil)

	var ix int = 0
	for cn := 0; cn < classes; cn++ {
		var r float64 = 0
		var t float64 = float64(cn * 4)

		for {

			if r > 1 && t > float64((cn+1)*4) {
				break
			}

			ran_t := t + (-1+rand.Float64()*(1-(-1)))*0.2

			X.Set(ix, 0, r*math.Sin(ran_t*2.5))
			X.Set(ix, 1, r*math.Cos(ran_t*2.5))
			y.Set(ix, 0, float64(cn))

			r += 1 / float64((points - 1))
			t += 4 / float64((points - 1))

			ix++
		}
	}
	return X, y
}
