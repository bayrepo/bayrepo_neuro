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

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
	"github.com/go-echarts/go-echarts/v2/types"
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
	Dat    string
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

func toDateFromInternal(d int) (string, int, int, int) {
	date1 := Date(1990, 1, 1)
	date2 := date1.Add(time.Duration(time.Hour * 24 * time.Duration(d)))
	return date2.Format("02.01.2006"), date2.Day(), int(date2.Month()), date2.Year()
}

func genrateLineItemsNew() map[int]Item {
	data := make(map[int]Item)
	xmlFile, err := os.Open("data.xml")
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

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

	interval := keys[len(keys)-1] - keys[0]

	var old_v float64
	old_v = 0.0
	less_key := keys[0]
	premassive := make(map[int]Item)
	for i := 0; i < interval; i++ {
		if val, ok := data[less_key+i]; ok {
			old_v = val.V
		}
		var tmp Item
		dt_s, D, M, Y := toDateFromInternal(less_key + i)
		tmp.V = old_v
		tmp.D = D
		tmp.M = M
		tmp.Y = Y
		tmp.Dat = dt_s
		premassive[less_key+i] = tmp
	}

	return premassive
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

func genrateLineItemsNeuro(input map[int]Item) map[int]Item {
	premassive := make(map[int]Item)

	for k := range input {
		var tmp Item
		D_in := float64(input[k].D) / 31.0
		M_in := float64(input[k].M) / 12.0

		inputs := []float64{D_in, M_in}

		res, outputs, err := webSendDataToNet(2, inputs)
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
			tmp.V = 50.0
		} else if fnd_ind == 2 {
			tmp.V = 5.0
		} else {
			tmp.V = 100.0
		}
		premassive[k] = tmp
	}

	return premassive
}

func httpserver(w http.ResponseWriter, _ *http.Request) {
	line := charts.NewLine()
	line.SetGlobalOptions(
		charts.WithInitializationOpts(opts.Initialization{Theme: types.ThemeWesteros}),
		charts.WithTitleOpts(opts.Title{
			Title:    "Currencies course",
			Subtitle: "Line chart rendered by the http server this time",
		}),
		charts.WithDataZoomOpts(opts.DataZoom{
			Type:  "slider",
			Start: 0,
			End:   100,
		}),
	)

	perm := genrateLineItemsNew()
	keys := make([]int, 0, len(perm))
	for k := range perm {
		keys = append(keys, k)
	}
	sort.Ints(keys)

	XNames := []string{}
	for _, k := range keys {
		XNames = append(XNames, perm[k].Dat)
	}
	YVal1 := []opts.LineData{}
	for _, k := range keys {
		YVal1 = append(YVal1, opts.LineData{Value: perm[k].V})
	}
	neur := genrateLineItemsNeuro(perm)
	YVal2 := []opts.LineData{}
	for _, k := range keys {
		YVal2 = append(YVal2, opts.LineData{Value: neur[k].V})
	}

	line.SetXAxis(XNames).
		AddSeries("Rubl", YVal1).
		AddSeries("Neur", YVal2).
		SetSeriesOptions(charts.WithLineChartOpts(opts.LineChart{Smooth: false}))
	line.Render(w)
}

func main() {
	http.HandleFunc("/", httpserver)
	http.ListenAndServe(":8081", nil)
}
