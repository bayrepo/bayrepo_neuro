package main

import (
	"bytes"
	"encoding/xml"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

type ValC struct {
	XMLName    xml.Name `xml:"ValCurs"`
	ID         string   `xml:"ID,attr"`
	DateRange1 string   `xml:"DateRange1,attr"`
	DateRange2 string   `xml:"DateRange2,attr"`
	name       string   `xml:"name,attr"`
	Records    []Record `xml:"Record"`
}

type Record struct {
	XMLName xml.Name `xml:"Record"`
	ID      string   `xml:"ID,attr"`
	Date    string   `xml:"Date,attr"`
	Nominal string   `xml:"Nominal"`
	Value   string   `xml:"Value"`
}

type Item struct {
	V      float64
	D      int
	M      int
	Y      int
	Change int
	Mass   float64
}

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

func main() {
	data := make(map[int]Item)
	args := os.Args
	if len(args) < 3 {
		fmt.Println("Too fee args. Usage: prg file netid")
		os.Exit(1)
	}
	xmlFile, err := os.Open(args[1])
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	netID, err := strconv.Atoi(args[2])
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Printf("Successfully Opened %s\n", args[1])
	defer xmlFile.Close()

	byteValue, _ := ioutil.ReadAll(xmlFile)

	var valc ValC

	xml.Unmarshal(byteValue, &valc)

	for i := 0; i < len(valc.Records); i++ {
		var tmp Item
		var p int
		n, _ := strconv.Atoi(valc.Records[i].Nominal)
		v, _ := strconv.ParseFloat(strings.Replace(valc.Records[i].Value, ",", ".", 1), 64)
		p, tmp.Y, tmp.M, tmp.D = translate(valc.Records[i].Date)
		tmp.V = v / float64(n)
		data[p] = tmp

	}

	keys := make([]int, 0, len(data))
	for k := range data {
		keys = append(keys, k)
	}
	sort.Ints(keys)

	change_days := float64(0.0)

	for i, k := range keys {
		if i > 0 {
			tmp := data[k]
			if data[k].V < data[keys[i-1]].V {
				change_days = 0.0
				tmp.Change = 2
				tmp.Mass = (data[keys[i-1]].V - data[k].V) / data[keys[i-1]].V
			} else if data[k].V > data[keys[i-1]].V {
				change_days = 0.0
				tmp.Change = 1
				tmp.Mass = (data[k].V - data[keys[i-1]].V) / data[k].V
			} else {
				change_days = change_days + 1.0
				tmp.Change = 0
				tmp.Mass = change_days / 365.0
			}
			data[k] = tmp
		}

	}

	for j := 0; j < 100; j = j + 1 {
		fmt.Printf("Learning iteration %d\n", j)
		for _, k := range keys {

			D_in := float64(data[k].D) / 31.0
			M_in := float64(data[k].M) / 12.0

			inputs := []float64{D_in, M_in}
			outputs := []float64{0.0, 0.0, 0.0}
			switch data[k].Change {
			case 0:
				outputs[0] = data[k].Mass
			case 1:
				outputs[1] = data[k].Mass
			default:
				outputs[2] = data[k].Mass
			}

			res, err := webSendTrainToNet(netID, inputs, outputs)
			if res == 0 {
				fmt.Println(err)
				os.Exit(1)
			}
		}
	}

}
