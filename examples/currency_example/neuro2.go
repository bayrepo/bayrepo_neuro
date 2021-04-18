package main

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

func Date(year, month, day int) time.Time {
	return time.Date(year, time.Month(month), day, 0, 0, 0, 0, time.UTC)
}

func translate(dst string) (int, int, int, int) {
	s := strings.Split(dst, ".")
	date1 := Date(1990, 1, 1)
	y, _ := strconv.Atoi(s[2])
	m, _ := strconv.Atoi(s[1])
	d, _ := strconv.Atoi(s[0])
	date2 := Date(y, m, d)
	var days int
	days = int(date2.Sub(date1).Hours() / 24)
	return days, y, m, d
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

func main() {
	args := os.Args
	if len(args) < 3 {
		fmt.Println("Too fee args. Usage: prg date netid")
		os.Exit(1)
	}

	//dd.mm.yyyy
	_, _, M, D := translate(args[1])

	netID, err := strconv.Atoi(args[2])
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	D_in := float64(D) / 31.0
	M_in := float64(M) / 12.0

	inputs := []float64{D_in, M_in}

	res, outputs, err := webSendDataToNet(netID, inputs)
	if res == 0 {
		fmt.Println(err)
		os.Exit(1)
	}

	fnd_ind := int(0)
	max := outputs[0]
	for ind := 0; ind < len(outputs); ind = ind + 1 {
		if max < outputs[ind] {
			max = outputs[ind]
			fnd_ind = ind
		}
	}

	if fnd_ind == 0 {
		fmt.Println("Not changed", outputs)
	} else if fnd_ind == 2 {
		fmt.Println("Got down", outputs)
	} else {
		fmt.Println("Got up", outputs)
	}

}
