package main

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

func webSendTrainToNet(NetID int, inputs []float64, outputs []float64) (int, error) {
	formData := url.Values{
		"ajax": {"y"},
	}
	var str_name string
	var val_name string
	for i, k := range inputs {
		str_name = fmt.Sprintf("tin2_%d", i)
		val_name = fmt.Sprintf("%f", k)
		formData.Set(str_name, val_name)
	}
	for i, k := range outputs {
		str_name = fmt.Sprintf("tine2_%d", i)
		val_name = fmt.Sprintf("%f", k)
		formData.Set(str_name, val_name)
	}

	client := &http.Client{
		Timeout: time.Duration(5 * time.Second),
	}

	req, err := http.NewRequest("POST", fmt.Sprintf("http://127.0.0.1:10111/addtrn/%d", NetID), bytes.NewBufferString(formData.Encode()))
	req.Header.Set("cache-control", "no-cache")
	if err != nil {
		return 0, err
	}

	resp, err := client.Do(req)

	if err != nil {
		return 0, err
	}

	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)

	if err != nil {
		return 0, err
	}

	string_body := string(body[:])

	if strings.Contains(string_body, "TYPE:TRAINING") != true {
		return 0, errors.New("Incorrect result")
	}

	return 1, nil
}

func webSendDataToNet(NetID int, inputs []float64) (int, []float64, error) {
	outputs := []float64{}
	formData := url.Values{
		"ajax": {"y"},
	}
	var str_name string
	var val_name string
	for i, k := range inputs {
		str_name = fmt.Sprintf("fin2_%d", i)
		val_name = fmt.Sprintf("%f", k)
		formData.Set(str_name, val_name)
	}

	client := &http.Client{
		Timeout: time.Duration(5 * time.Second),
	}

	req, err := http.NewRequest("POST", fmt.Sprintf("http://127.0.0.1:10111/addinp/%d", NetID), bytes.NewBufferString(formData.Encode()))
	req.Header.Set("cache-control", "no-cache")
	if err != nil {
		return 0, outputs, err
	}

	resp, err := client.Do(req)

	if err != nil {
		return 0, outputs, err
	}

	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)

	if err != nil {
		return 0, outputs, err
	}

	string_body := string(body[:])

	if strings.Contains(string_body, "TYPE:RESULT") != true {
		return 0, outputs, errors.New("Incorrect result")
	}
	result := strings.Split(string_body, "\n")
	new_outputs := make([]float64, len(result)-2)

	for i, _ := range new_outputs {
		new_outputs[i] = 0.0
	}

	for _, v := range result[1:] {
		data := strings.Split(v, ":")
		if len(data) > 1 {
			ind, _ := strconv.Atoi(data[0])
			val, _ := strconv.ParseFloat(data[1], 64)
			new_outputs[ind] = val
		}
	}

	return 1, new_outputs, nil
}

func neededField2() ([][]int, [][]int) {
	fld := [][]int{
		{1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		{1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 1, 1, 1},
		{0, 0, 0, 0, 0, 0, 0, 1, 1, 1},
		{0, 0, 0, 0, 0, 0, 0, 1, 1, 1},
	}

	return fld, fld
}

func neededField() ([][]int, [][]int) {
	fld := [][]int{
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	}

	lns := rand.Intn(5) + 2
	inln := rand.Intn(5) + 2

	x := rand.Intn(len(fld[0]) - 1)
	y := rand.Intn(len(fld) - 1)
	if (x + inln) >= len(fld[0]) {
		x = len(fld[0]) - inln - 1
	}
	if (y + lns) >= len(fld) {
		y = len(fld) - lns - 1
	}

	for ind := 0; ind < lns; ind = ind + 1 {
		for ind2 := 0; ind2 < inln; ind2 = ind2 + 1 {
			fld[y+ind][x+ind2] = 1
		}
	}

	fld_check := [][]int{}
	for i := 0; i < len(fld); i = i + 1 {
		a := []int{}
		for j := 0; j < len(fld[i]); j = j + 1 {
			a = append(a, fld[i][j])
		}
		fld_check = append(fld_check, a)
	}

	for k := 0; k < 8; k = k + 1 {
		x := rand.Intn(len(fld[0]) - 1)
		y := rand.Intn(len(fld) - 1)
		fld[y][x] = 1
	}

	return fld, fld_check
}

// LU BU RU
// LB XX RB
// LD BD RD
type ItemData struct {
	Weight float64
	LU     float64
	BU     float64
	RU     float64
	LB     float64
	RB     float64
	LD     float64
	BD     float64
	RD     float64
}

func convertItemData(dat ItemData) [9]float64 {
	return [9]float64{dat.Weight, dat.LU, dat.BU, dat.RU, dat.LB, dat.RB, dat.LD, dat.BD, dat.RD}
}

func translateFieldToNets(input [][]int) []float64 {
	result := []float64{}
	preresult := []ItemData{}
	for _, k1 := range input {
		for _, k2 := range k1 {
			tmp := ItemData{}
			if k2 > 0 {
				tmp.Weight = 0.99
			}
			preresult = append(preresult, tmp)
		}
	}
	for k1 := 0; k1 < len(input); k1 = k1 + 1 {
		for k2 := 0; k2 < len(input[k1]); k2 = k2 + 1 {
			if k1-1 >= 0 {
				if k2-1 >= 0 {
					if input[k1-1][k2-1] > 0 {
						preresult[k1*len(input[k1])+k2].LU = 0.99
					} else {
						preresult[k1*len(input[k1])+k2].LU = 0.0
					}
				} else {
					preresult[k1*len(input[k1])+k2].LU = 0.0
				}
				if input[k1-1][k2] > 0 {
					preresult[k1*len(input[k1])+k2].BU = 0.99
				} else {
					preresult[k1*len(input[k1])+k2].BU = 0.0
				}
				if k2+1 < len(input[k1]) {
					if input[k1-1][k2+1] > 0 {
						preresult[k1*len(input[k1])+k2].RU = 0.99
					} else {
						preresult[k1*len(input[k1])+k2].RU = 0.0
					}
				}
			} else {
				preresult[k1*len(input[k1])+k2].LU = 0.0
				preresult[k1*len(input[k1])+k2].BU = 0.0
				preresult[k1*len(input[k1])+k2].RU = 0.0
			}

			if k2-1 >= 0 {
				if input[k1][k2-1] > 0 {
					preresult[k1*len(input[k1])+k2].LB = 0.99
				} else {
					preresult[k1*len(input[k1])+k2].LB = 0.0
				}
			} else {
				preresult[k1*len(input[k1])+k2].LB = 0.0
			}
			if k2+1 < len(input[k1]) {
				if input[k1][k2+1] > 0 {
					preresult[k1*len(input[k1])+k2].RB = 0.99
				} else {
					preresult[k1*len(input[k1])+k2].RB = 0.0
				}
			}

			if k1+1 < len(input) {
				if k2-1 >= 0 {
					if input[k1+1][k2-1] > 0 {
						preresult[k1*len(input[k1])+k2].LD = 0.99
					} else {
						preresult[k1*len(input[k1])+k2].LD = 0.0
					}
				} else {
					preresult[k1*len(input[k1])+k2].LD = 0.0
				}
				if input[k1+1][k2] > 0 {
					preresult[k1*len(input[k1])+k2].BD = 0.99
				} else {
					preresult[k1*len(input[k1])+k2].BD = 0.0
				}
				if k2+1 < len(input[k1]) {
					if input[k1+1][k2+1] > 0 {
						preresult[k1*len(input[k1])+k2].RD = 0.99
					} else {
						preresult[k1*len(input[k1])+k2].RD = 0.0
					}
				}
			} else {
				preresult[k1*len(input[k1])+k2].LD = 0.0
				preresult[k1*len(input[k1])+k2].BD = 0.0
				preresult[k1*len(input[k1])+k2].RD = 0.0
			}
		}
	}

	for _, k := range preresult {
		res := convertItemData(k)
		result = append(result, res[:]...)
	}

	return result
}

func translateFieldToTrainNets(input [][]int) []float64 {
	result := []float64{}
	for _, k1 := range input {
		for _, k2 := range k1 {
			if k2 > 0 {
				result = append(result, 0.99)
			} else {
				result = append(result, 0.0)
			}
		}
	}
	return result
}

func printResult(outputs []float64, inputs [][]int, check [][]int) {
	fmt.Println("Inp")
	for ind := 0; ind < len(inputs); ind = ind + 1 {
		for ind2 := 0; ind2 < len(inputs[ind]); ind2 = ind2 + 1 {
			if inputs[ind][ind2] > 0 {
				fmt.Print("x")
			} else {
				fmt.Print("o")
			}
		}
		fmt.Println("")
	}
	fmt.Println("Check")
	for ind := 0; ind < len(check); ind = ind + 1 {
		for ind2 := 0; ind2 < len(check[ind]); ind2 = ind2 + 1 {
			if check[ind][ind2] > 0 {
				fmt.Print("x")
			} else {
				fmt.Print("o")
			}
		}
		fmt.Println("")
	}
	fmt.Println("Net")
	for ind := 0; ind < len(inputs); ind = ind + 1 {
		for ind2 := 0; ind2 < len(inputs[ind]); ind2 = ind2 + 1 {
			if outputs[ind*len(inputs[ind])+ind2] > 0.8 {
				fmt.Print("x")
			} else {
				fmt.Print("o")
			}
		}
		fmt.Println("")
	}
}

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	netID := 6

	for ind := 0; ind < 50; ind = ind + 1 {
		datInp, datCheck := neededField()
		inputs := translateFieldToNets(datInp)
		tr_outputs := translateFieldToTrainNets(datCheck)
		for ind1 := 0; ind1 < 100; ind1 = ind1 + 1 {
			res, err := webSendTrainToNet(netID, inputs, tr_outputs)
			if res == 0 {
				fmt.Println(err)
				os.Exit(1)
			}
		}
		fmt.Println(ind)
	}

	datInp, datCheck := neededField()
	inputs := translateFieldToNets(datInp)

	res, outputs, err := webSendDataToNet(netID, inputs)
	if res == 0 {
		fmt.Println(err)
		os.Exit(1)
	}

	printResult(outputs, datInp, datCheck)
	fmt.Println(outputs)

}
