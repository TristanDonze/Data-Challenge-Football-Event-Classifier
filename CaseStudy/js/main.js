const ctx = {
    training_data_loaded: false,
    testing_data_loaded: false,
    WIDTH: 600,
    HEIGHT: 600,
    AXIS_HEIGHT: 30,
};



function create_viz() {
    console.log("Using D3 v" + d3.version);

    d3.select("body").on("keydown", function(event, d){handleKeyEvent(event);});

    let temporal_graph1 = d3.select("main section.temporal-graphs .graphs-wrapper").append("svg").attr("id", "temporal-graph1");
    temporal_graph1.attr("width", ctx.WIDTH).attr("height", ctx.HEIGHT);

    let temporal_graph2 = d3.select("main section.temporal-graphs .graphs-wrapper").append("svg").attr("id", "temporal-graph2");
    temporal_graph2.attr("width", ctx.WIDTH).attr("height", ctx.HEIGHT);

    let tweet_graph1 = d3.select("main section.tweet-graphs .graphs-wrapper").append("svg").attr("id", "tweet-graph1");
    tweet_graph1.attr("width", ctx.WIDTH).attr("height", ctx.HEIGHT);

    let tweet_graph2 = d3.select("main section.tweet-graphs .graphs-wrapper").append("svg").attr("id", "tweet-graph2");
    tweet_graph2.attr("width", ctx.WIDTH).attr("height", ctx.HEIGHT);

    let full_temporal_graph1 = d3.select("main section.full-temporal-graphs .graphs-wrapper").append("svg").attr("id", "fulll-temporal-graph1");
    full_temporal_graph1.attr("width", ctx.WIDTH).attr("height", ctx.HEIGHT);

    let full_temporal_graph2 = d3.select("main section.full-temporal-graphs .graphs-wrapper").append("svg").attr("id", "fulll-temporal-graph2");
    full_temporal_graph2.attr("width", ctx.WIDTH).attr("height", ctx.HEIGHT);

    let full_temporal_graph3 = d3.select("main section.full-temporal-graphs .graphs-wrapper").append("svg").attr("id", "fulll-temporal-graph3");
    full_temporal_graph3.attr("width", ctx.WIDTH).attr("height", ctx.HEIGHT);

    let full_temporal_graph4 = d3.select("main section.full-temporal-graphs .graphs-wrapper").append("svg").attr("id", "fulll-temporal-graph4");
    full_temporal_graph4.attr("width", ctx.WIDTH).attr("height", ctx.HEIGHT);

    // d3.select("#main svg").append("g").attr("id", "us_map");
    // d3.select("#main svg").append("g").attr("id", "routeG");
    // d3.select("#main svg").append("g").attr("id", "airportG");
    load_data();
};

function load_data(){
    const test_files = [
        "data/eval_tweets/GermanyGhana32.csv",
        "data/eval_tweets/GermanySerbia2010.csv",
        "data/eval_tweets/GreeceIvoryCoast44.csv",
        "data/eval_tweets/NetherlandsMexico64.csv",
    ]

    const train_files = [
        "data/train_tweets/ArgentinaBelgium72.csv",
        "data/train_tweets/ArgentinaGermanyFinal77.csv",
        "data/train_tweets/AustraliaNetherlands29.csv",
        "data/train_tweets/AustraliaSpain34.csv",
        "data/train_tweets/BelgiumSouthKorea59.csv",
        "data/train_tweets/CameroonBrazil36.csv",
        "data/train_tweets/FranceGermany70.csv",
        "data/train_tweets/FranceNigeria66.csv",
        "data/train_tweets/GermanyAlgeria67.csv",
        "data/train_tweets/GermanyBrazil74.csv",
        "data/train_tweets/GermanyUSA57.csv",
        "data/train_tweets/HondurasSwitzerland54.csv",
        "data/train_tweets/MexicoCroatia37.csv",
        "data/train_tweets/NetherlandsChile35.csv",
        "data/train_tweets/PortugalGhana58.csv",
        "data/train_tweets/USASlovenia2010.csv"
    ]

    // csv files
    Promise.all(train_files.map(d => d3.csv(d)))
        .then(data => {
            mapped_data = {}
            train_files.forEach((d, i) => {
                mapped_data[d] = data[i];
            });

            ctx.train_data = mapped_data;
            ctx.training_data_loaded = true;
            create_graphs()
        })
        .catch(error => {
            console.error("Error loading data:", error);
            throw error;
        });
    
    Promise.all(test_files.map(d => d3.csv(d)))
        .then(data => {
            mapped_data = {}
            test_files.forEach((d, i) => {
                mapped_data[d] = data[i];
            });

            ctx.test_data = mapped_data;
            ctx.testing_data_loaded = true;
            create_graphs()
        })
        .catch(error => {
            console.error("Error loading data:", error);
            throw error;
        });
};

function smoothData(aggregated_array, ecart) {
    return aggregated_array.map((current, index, array) => {
        // Define the range of periods to consider for smoothing
        let start = Math.max(0, index - ecart);  // Prevent going negative
        let end = Math.min(array.length - 1, index + ecart);  // Prevent going out of bounds

        // Get the values of the periods in the range [start, end]
        let valuesInRange = array.slice(start, end + 1).map(d => d.y);

        // Calculate the mean of the selected periods
        let smoothedMean = d3.mean(valuesInRange);

        // Return the smoothed data
        return { x: current.x, y: smoothedMean };
    });
}

function create_graphs() {
    if (!(ctx.training_data_loaded && ctx.testing_data_loaded)) {
        return;
    }
    d3.select("#loading-message").remove();
    console.log("Data loaded successfully");
    populate_graphs();
}

function populate_graphs() {
    const temporal_graph1 = d3.select("#temporal-graph1");
    temporal_graph1.append("text")
        .attr("x", ctx.WIDTH / 2)
        .attr("y", 20)            
        .attr("text-anchor", "middle")
        .attr("font-size", "20px")     
        .text("Nombre de tweets en % par période");


    const temporal_graph2 = d3.select("#temporal-graph2");
    temporal_graph2.append("text")
        .attr("x", ctx.WIDTH / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .attr("font-size", "20px")
        .text("Taille moyenne des tweets en % par période");

    const tweet_graph1 = d3.select("#tweet-graph1");
    tweet_graph1.append("text")
        .attr("x", ctx.WIDTH / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .attr("font-size", "20px")
        .text("Nombre de tweets / Taille moyenne des tweets");

    const tweet_graph2 = d3.select("#tweet-graph2");
    tweet_graph2.append("text")
        .attr("x", ctx.WIDTH / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .attr("font-size", "20px")
        .text("Var Pourcentage de chaque lettre par période (Noir = 10%, Blanc = -10%)");

    const full_temporal_graph1 = d3.select("#fulll-temporal-graph1");
    full_temporal_graph1.append("text")
        .attr("x", ctx.WIDTH / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .attr("font-size", "20px")
        .text("Répartition des events par période (Smoothed)");

    const full_temporal_graph2 = d3.select("#fulll-temporal-graph2");
    full_temporal_graph2.append("text")
        .attr("x", ctx.WIDTH / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .attr("font-size", "20px")
        .text("AVG Répartition des events par période (Smoothed)");

    const full_temporal_graph3 = d3.select("#fulll-temporal-graph3");
    full_temporal_graph3.append("text")
        .attr("x", ctx.WIDTH / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .attr("font-size", "20px")
        .text("Events par période folded by sin");

    const full_temporal_graph4 = d3.select("#fulll-temporal-graph4");
    full_temporal_graph4.append("text")
        .attr("x", ctx.WIDTH / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .attr("font-size", "20px")
        .text("AVG Répartition des events par période folded by sin");
    

    const domainX = [0, 180]
    const domainX2 = [0, 100]
    const domainY = [0, 100]

    const ECART_SMOOTHING = 2;

    const xScale = d3.scaleLinear().domain(domainX).range([ctx.AXIS_HEIGHT, ctx.WIDTH - ctx.AXIS_HEIGHT])
    const xScale2 = d3.scaleLinear().domain(domainX2).range([ctx.AXIS_HEIGHT, ctx.WIDTH - ctx.AXIS_HEIGHT])
    const xScale3 = d3.scaleLinear().domain([0, 40]).range([ctx.AXIS_HEIGHT, ctx.WIDTH - ctx.AXIS_HEIGHT])
    const yScale = d3.scaleLinear().domain(domainY).range([ctx.HEIGHT - ctx.AXIS_HEIGHT, ctx.AXIS_HEIGHT])
    const yScale2 = d3.scaleLinear().domain([0, 1.6]).range([ctx.HEIGHT - ctx.AXIS_HEIGHT, ctx.AXIS_HEIGHT])
    const colorScale = d3.scaleOrdinal().domain([0, 1]).range(["red", "green"]);
    const colorScale2 = d3.scaleLinear().domain([-10, 10]).range(["white", "black"])

    const xAxis = d3.axisBottom(xScale);
    const yAxis = d3.axisLeft(yScale);
    const xAxis2 = d3.axisBottom(xScale2);
    const xAxis3 = d3.axisBottom(xScale3);
    const yAxis2 = d3.axisLeft(yScale2);

    temporal_graph1.append("g").attr("class", "x-axis").call(xAxis).attr("transform", `translate(0, ${ctx.HEIGHT - ctx.AXIS_HEIGHT})`);
    temporal_graph1.append("g").attr("class", "y-axis").call(yAxis).attr("transform", `translate(${ctx.AXIS_HEIGHT}, 0)`);

    temporal_graph2.append("g").attr("class", "x-axis").call(xAxis).attr("transform", `translate(0, ${ctx.HEIGHT - ctx.AXIS_HEIGHT})`);
    temporal_graph2.append("g").attr("class", "y-axis").call(yAxis).attr("transform", `translate(${ctx.AXIS_HEIGHT}, 0)`);

    tweet_graph1.append("g").attr("class", "x-axis").call(xAxis2).attr("transform", `translate(0, ${ctx.HEIGHT - ctx.AXIS_HEIGHT})`);
    tweet_graph1.append("g").attr("class", "y-axis").call(yAxis).attr("transform", `translate(${ctx.AXIS_HEIGHT}, 0)`);

    tweet_graph2.append("g").attr("class", "x-axis").call(xAxis).attr("transform", `translate(0, ${ctx.HEIGHT - ctx.AXIS_HEIGHT})`);

    full_temporal_graph1.append("g").attr("class", "x-axis").call(xAxis).attr("transform", `translate(0, ${ctx.HEIGHT - ctx.AXIS_HEIGHT})`);
    full_temporal_graph1.append("g").attr("class", "y-axis").call(yAxis2).attr("transform", `translate(${ctx.AXIS_HEIGHT}, 0)`);

    full_temporal_graph2.append("g").attr("class", "x-axis").call(xAxis).attr("transform", `translate(0, ${ctx.HEIGHT - ctx.AXIS_HEIGHT})`);
    full_temporal_graph2.append("g").attr("class", "y-axis").call(yAxis2).attr("transform", `translate(${ctx.AXIS_HEIGHT}, 0)`);

    full_temporal_graph3.append("g").attr("class", "x-axis").call(xAxis3).attr("transform", `translate(0, ${ctx.HEIGHT - ctx.AXIS_HEIGHT})`);
    full_temporal_graph3.append("g").attr("class", "y-axis").call(yAxis2).attr("transform", `translate(${ctx.AXIS_HEIGHT}, 0)`);

    full_temporal_graph4.append("g").attr("class", "x-axis").call(xAxis3).attr("transform", `translate(0, ${ctx.HEIGHT - ctx.AXIS_HEIGHT})`);
    full_temporal_graph4.append("g").attr("class", "y-axis").call(yAxis2).attr("transform", `translate(${ctx.AXIS_HEIGHT}, 0)`);

    let legend = d3.select("main").append("div").attr("class", "legend");
    
    ctx.wanted_libelle = new Set();

    let data_full_temporal2 = {}
    let data_full_temporal4 = {}

    const gradient = full_temporal_graph1.append("defs")
        .append("linearGradient")
        .attr("id", "line-gradient")
        .attr("x1", "0%")
        .attr("y1", "0%")  // Start gradient at the top
        .attr("x2", "0%")
        .attr("y2", "100%");  // Gradient will go vertically (along y)
    
    gradient.append("stop")
        .attr("offset", "40%")
        .attr("stop-color", "green");  // Color at the top (for y = 0)
    
    gradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", "red");  // Color at the bottom (for y = 1)

    // transform time to a sin wave to curve it on itself and allow more similarity
    function timefold(time) {
        // output is between 0 and 40 (cuz of the x axis)
        x = Math.sin(time / 180 * Math.PI) * 40; //quite good but the start is messy
        // cylce = time % 60
        // return Math.abs(Math.sin(Math.abs(Math.sin(cylce / 60 * Math.PI) * 2 * Math.PI))* 40)
        return Math.abs(x);
    }

    Object.keys(ctx.train_data).forEach((libelle) => {
        let clear_libelle = libelle.split("/").pop().split(".")[0];
        
        legend.append("div").attr("class", "legend-item")
            .text(clear_libelle)
            .style("text-decoration", "line-through")
            .on("click", function() {
                if (this.style.textDecoration === "line-through") {
                    this.style.textDecoration = "none"
                    ctx.wanted_libelle.add(this.textContent)
                    list_data = {}
                    ctx.wanted_libelle.forEach((libelle) => {
                        libelle = "data/train_tweets/" + libelle + ".csv";
                        list_data[libelle] = ctx.train_data[libelle];
                    })
                    d3.selectAll("g.line-group").remove();
                    draw_data(list_data, xScale, yScale, colorScale, colorScale2)
                    
                } else {
                    this.style.textDecoration = "line-through"
                    ctx.wanted_libelle.delete(this.textContent)
                    list_data = {}
                    ctx.wanted_libelle.forEach((libelle) => {
                        libelle = "data/train_tweets/" + libelle + ".csv";
                        list_data[libelle] = ctx.train_data[libelle];
                    })
                    d3.selectAll("g.line-group").remove();
                    draw_data(list_data, xScale, yScale, colorScale, colorScale2)
                }
            })

        
        // periodID, EventType
        let aggregated_data = d3.rollup(ctx.train_data[libelle], v => d3.mean(v, d => d.EventType), d => parseInt(d.PeriodID));
        let aggregated_data_array = Array.from(aggregated_data, ([key, value]) => ({x: parseInt(key), y: value}));
        aggregated_data = smoothData(aggregated_data_array, ECART_SMOOTHING + 1);
        aggregated_data = smoothData(aggregated_data, ECART_SMOOTHING - 1);

        aggregated_data.forEach((d) => {
            if (!data_full_temporal2[d.x]) {
                data_full_temporal2[d.x] = []
            }
            data_full_temporal2[d.x].push(d.y);
        })

        full_temporal_graph1.append("g")
            .attr("id", "line")
            .selectAll("path")
            .data([aggregated_data])
            .enter()
            .append("path")
            .attr("d", d3.line()
                .x(d => xScale(d.x))
                .y(d => yScale2(d.y))
                .curve(d3.curveCardinal)
            )
            .attr("stroke", "url(#line-gradient)")
            .attr("stroke-width", 2)
            .attr("fill", "none")

        
        aggregated_data = aggregated_data.map(d => ({x: timefold(d.x), y: d.y}));
        aggregated_data.forEach((d) => {
            if (!data_full_temporal4[d.x]) {
                data_full_temporal4[d.x] = []
            }
            data_full_temporal4[d.x].push(d.y);
        })

        full_temporal_graph3.append("g")
            .attr("id", "line")
            .selectAll("path")
            .data([aggregated_data])
            .enter()
            .append("path")
            .attr("d", d3.line()
                .x(d => xScale3(d.x))
                .y(d => yScale2(d.y))
                .curve(d3.curveCardinal)
            )
            .attr("stroke", "url(#line-gradient)")
            .attr("stroke-width", 2)
            .attr("fill", "none")
    })

    data_full_temporal2 = Object.entries(data_full_temporal2).map(([key, value]) => ({x: parseInt(key), y: Math.pow(d3.mean(value), 2.5)}));
    full_temporal_graph2.append("g")
        .attr("id", "line")
        .selectAll("path")
        .data([data_full_temporal2])
        .enter()
        .append("path")
        .attr("d", d3.line()
            .x(d => xScale(d.x))
            .y(d => yScale2(d.y))
            .curve(d3.curveCardinal)
        )
        .attr("stroke", "url(#line-gradient)")
        .attr("stroke-width", 2)
        .attr("fill", "none")

    // draw a line at the avg
    let avg = d3.mean(data_full_temporal2, d => d.y);
    full_temporal_graph2.append("line")
        .attr("x1", xScale(0))
        .attr("y1", yScale2(avg))
        .attr("x2", xScale(180))
        .attr("y2", yScale2(avg))
        .attr("stroke", "black")
        .attr("stroke-width", 1)
        .style("stroke-dasharray", ("3, 3"));

    data_full_temporal4 = Object.entries(data_full_temporal4).map(([key, value]) => ({x: parseInt(key), y: d3.mean(value)}));
    // regroup by x, avg(y)
    data_full_temporal4 = Array.from(d3.rollup(data_full_temporal4, v => d3.mean(v, d => d.y), d => d.x), ([key, value]) => ({x: parseInt(key), y: value})).sort((a, b) => a.x - b.x);
    console.log(data_full_temporal4)
    full_temporal_graph4.append("g")
        .attr("id", "line")
        .selectAll("path")
        .data([data_full_temporal4])
        .enter()
        .append("path")
        .attr("d", d3.line()
            .x(d => xScale3(d.x))
            .y(d => yScale2(d.y))
            .curve(d3.curveCardinal)
        )
        .attr("stroke", "url(#line-gradient)")
        .attr("stroke-width", 2)
        .attr("fill", "none")

    // draw a line at the avg
    avg = d3.mean(data_full_temporal4, d => d.y);
    full_temporal_graph4.append("line")
        .attr("x1", xScale3(0))
        .attr("y1", yScale2(avg))
        .attr("x2", xScale3(40))
        .attr("y2", yScale2(avg))
        .attr("stroke", "black")
        .attr("stroke-width", 1)
        .style("stroke-dasharray", ("3, 3"));
}

function draw_data(list_data, xScale, yScale, colorScale, colorScale2) {
    const temporal_graph1 = d3.select("#temporal-graph1");
    const temporal_graph2 = d3.select("#temporal-graph2");
    const tweet_graph1 = d3.select("#tweet-graph1");
    const tweet_graph2 = d3.select("#tweet-graph2");

    Object.keys(list_data).forEach((libelle, i) => {
        const data = list_data[libelle];
        let clear_libelle = libelle.split("/").pop().split(".")[0];

        // aggregate data: count number of tweets per PeriodID (1 tweet = 1 row)
        let aggregated_data = d3.rollup(data, v => v.length, d => parseInt(d.PeriodID));
        let max_y = d3.max(Array.from(aggregated_data.values()));
        const aggregated_data_array = Array.from(aggregated_data, ([key, value]) => ({x: parseInt(key), y: value / max_y * 100}));
        aggregated_data = Object.fromEntries(aggregated_data_array.map(d => [d.x, d.y]));

        // labels :> PeriodID, parseInt(AVG(EventType))
        const labels_rollup = d3.rollup(
            data,
            v => d3.mean(v, d => d.EventType),  // Calcule la moyenne de EventType pour chaque PeriodID
            d => parseInt(d.PeriodID)                    // Regroupe par PeriodID
        );
        const labels = Object.fromEntries(labels_rollup);

        temporal_graph1.append("g")
            .attr("class", "line-group")
            .attr("id", clear_libelle)
            .selectAll("circle")
            .data(aggregated_data_array)
            .enter()
            .append("circle")
            .attr("cx", d => xScale(d.x))
            .attr("cy", d => yScale(d.y))
            .attr("r", 4)
            .attr("value", d => typeof(d.x))
            .attr("fill", (d) => colorScale(labels[d.x]))
            .style("opacity", 0.6)
            .append("title")
                .text(d => `PeriodID: ${d.x}`);

        // grah2.2 we put all the tweets, X is the PeriodID, Y is the Tweet.lenght
        function sum(array) {
            let tot = 0;
            array.forEach((d) => {
                tot += d;
            });
            return tot;
        }

        let aggregated_data2 = {}
        data.forEach((d) => {
            period = parseInt(d.PeriodID);
            if (!aggregated_data2[period]) {
                aggregated_data2[period] = []
            }
            aggregated_data2[period].push(d.Tweet.length);
        });

        let aggregated_data_array2 = Object.entries(aggregated_data2).map(([key, value]) => ({x: key, y: sum(value)}));
        max_y = d3.max(aggregated_data_array2, d => d.y);
        aggregated_data_array2 = aggregated_data_array2.map(d => ({x: d.x, y: d.y / max_y * 100}));

        temporal_graph2.append("g")
            .attr("class", "line-group")
            .attr("id", clear_libelle)
            .selectAll("circle")
            .data(aggregated_data_array2)
            .enter()
            .append("circle")
            .attr("cx", d => xScale(d.x))
            .attr("cy", d => yScale(d.y))
            .attr("r", 4)
            .attr("value", d => typeof(d.x))
            .attr("fill", (d) => colorScale(labels[d.x]))
            .style("opacity", 0.6)
            .append("title")
                .text(d => `PeriodID: ${d.x}`);


            // tweet graph: nb tweet / length moyen des tweets
            let aggregated_data3 = []
            Object.keys(aggregated_data2).forEach((period) => {
                aggregated_data3.push({x: aggregated_data[period], y: sum(aggregated_data2[period]) / max_y * 100, z: period});
            });

            tweet_graph1.append("g")
                .attr("class", "line-group")
                .attr("id", clear_libelle)
                .selectAll("circle")
                .data(aggregated_data3)
                .enter()
                .append("circle")
                .attr("cx", d => yScale(d.x))
                .attr("cy", d => yScale(d.y))
                .attr("r", 4)
                .attr("fill", (d) => colorScale(labels[d.z]))
                .style("opacity", 0.6)
                .append("title")
                    .text(d => `PeriodID: ${d.z}`);

            // tweet graph 2: avg % of letters of tweets by period
            let aggregated_data4 = []
            all_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ!?/#@.,;:".split("");
            all_letters = Array.from(["Event", "Tot", ...all_letters]);

            letters_avgs = {}

            data.forEach((d) => {
                period = parseInt(d.PeriodID);
                if (!aggregated_data4[period]) {
                    aggregated_data4[period] = {}
                }
                d.Tweet.split("").forEach((letter) => {
                    letter = letter.toUpperCase();
                    if (!aggregated_data4[period][letter]) {
                        aggregated_data4[period][letter] = 0;
                    }
                    aggregated_data4[period][letter] += 1;
                });

                all_letters.forEach((letter) => {
                    if (!letters_avgs[letter]) {
                        letters_avgs[letter] = 0;
                    }
                    letters_avgs[letter] += aggregated_data4[period][letter] || 0;
                })
            })

            Object.keys(letters_avgs).forEach((letter) => {
                letters_avgs[letter] /= data.length;
            })
            
            let aggregated_data_array4 = []
            tot_percents = []
            Object.keys(aggregated_data4).forEach((period) => {
                let sum_letters = 0;
                all_letters.forEach((letter) => {
                    sum_letters += aggregated_data4[period][letter] || 0;
                })

                y_vector = []
                all_letters.forEach((letter) => {
                    nb_of_letter = (aggregated_data4[period][letter] || 0) - letters_avgs[letter] 
                    y_vector.push(nb_of_letter / sum_letters * 100);
                })
                aggregated_data_array4.push({x: period, y: y_vector});
                tot_percents.push(sum(y_vector) / all_letters.length);
            })
            tweet_graph2.append("g")
                .attr("class", "line-group")
                .attr("id", clear_libelle)

            const yScale2 = d3.scaleBand().domain(all_letters).range([ctx.AXIS_HEIGHT, ctx.HEIGHT - ctx.AXIS_HEIGHT]).padding(0.1);
            tweet_graph2.selectAll("g.y-axis").remove();
            tweet_graph2.append("g").attr("class", "y-axis").call(d3.axisLeft(yScale2)).attr("transform", `translate(${ctx.AXIS_HEIGHT}, 0)`);

            aggregated_data_array4.forEach((d) => {
                d.y.forEach((value, i) => {
                    tweet_graph2.select("g#" + clear_libelle)
                        .append("rect")
                        .attr("x", xScale(d.x))
                        .attr("y", yScale2(all_letters[i]))
                        .attr("width", 2)
                        .attr("height", 12)
                        .attr("fill", colorScale2(parseFloat(value)))
                        .attr("value", value)
                })
            })

            let seuils = d3.extent(tot_percents)
            seuils = [seuils[0], (seuils[0] + seuils[1]) / 1.6, seuils[1]]
            let scaleColor3 = d3.scaleLinear().domain(seuils).range(["red", "white", "green"]);
            Object.keys(aggregated_data4).forEach((period) => {
                tweet_graph2.select("g#" + clear_libelle)
                    .append("rect")
                    .attr("x", xScale(period))
                    .attr("y", yScale2("Tot"))
                    .attr("width", 2)
                    .attr("height", 12)
                    .attr("fill", scaleColor3(tot_percents[period]))
                    .append("title")
                        .text(`PeriodID: ${period}`);
            })
            
            Object.keys(aggregated_data4).forEach((period) => {
                tweet_graph2.select("g#" + clear_libelle)
                    .append("rect")
                    .attr("x", xScale(period))
                    .attr("y", yScale2("Event"))
                    .attr("width", 2)
                    .attr("height", 12)
                    .attr("fill", colorScale(labels[period]))
                    .append("title")
                        .text(`PeriodID: ${period}`);
            })

    });
}


