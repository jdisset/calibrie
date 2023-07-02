<style>
  .md-typeset h1,
  .md-content__button {
    display: none;
  }
</style>



<figure markdown>
   ![Calibry Logo](./assets/calibry.svg){width=200px}
</figure>
**Calibry** is a modular python library for the **analysis**, **unmixing**, and **calibration** of fluorescence flow cytometry data.
	
It implements multiple algorithms to turn fluorescence values into protein quantities, many diagnostics and quality assessment tools, and a number of
plotting and analysis functions. It is designed to be extensible, and can be used as a library or as a standalone command line tool.

!!! warning 
	Calibry is currently in **alpha**. It is usable, but the API is not yet stable, and there are still some rough edges.
	

## Installation

Calibry is still, for now, internal to the Weiss lab only and is not yet available on PyPI. To install it, clone this repository and run

```bash
pip install .
```


## Usage

Here is a simple example of how to use Calibry to perform linear compensation.


```python
import calibry as cal

linpipe = cal.Pipeline(
    [
        cal.LoadControlsFromCSV("controls.csv"),
        cal.LinearCompensation(),
    ]
)

linpipe.initialize()

# to plot spillover matrix and single-protein compensation:
linpipe.diagnostics("LinearCompensation")

# ... after loading observations:
abundances = linpipe.apply(observations)['abundances_AU']

```

## Architecture Overview

Calibry is built around the concept of **pipelines**, and the use of **context keys**. A pipeline is a sequence of **tasks** that produce and use data from other tasks.
This data is stored in a context dictionary under a documented, specific key, that can be used by other tasks.

This way, tasks can be designed to be relatively independent from each other, and can be reused in various combinations across different pipelines, as long as the context keys,
a.k.a "pipeline data" they need are provided by the previous tasks. 

An example of context keys would be `controls_values`, which is an array of fluorescence values produced by the `LoadControls` task, and used by most compensation tasks.

A list of all the context keys required (and produced) by a task can be found in the documentation of the task.

Calibry provides a number of tasks, which can be used to build pipelines and range from simple data loading to complex compensation and calibration algorithms. 
Many tasks provide a diagnostics method, which can be used to plot the results of the task and assess their quality.

As convenience, Calibry also provides a number of pre-built pipelines, which can be used as-is, or as a starting point for more complex pipelines.
Users can also write their own tasks, and use them to build their custom pipelines.

Pipeline can also be used for data analysis, and Calibry provides a number of analysis tasks.


### Initialization

The initialize method is called once, before the pipeline is applied to any data. It is used to initialize the tasks, and to populate the context dictionary with each task's produced init data. 

This, for example, is when the `LoadControls` task will load the controls from a CSV file, and store them in the context dictionary under the key `controls_values`, and when
the `LinearUnmixing` task will compute the spillover matrix from the controls, and store it in the context dictionary under the key `spillover_matrix`.

### Application

The apply method is designed to process an array, or dataframe, of observations. It is called once for each batch of observations, and uses a second context dictionary, the batch context, to store and retrieve data.

For example, the `LinearUnmixing` task will use the spillover matrix stored in the init context to compute the arbitraty units abundances of each protein in each observation, and store them in the batch context under the key `abundances_AU`.

Then, if present, the `MEFCalibration` task will use these abundances to standardize them to MEFs, and store the result in the batch context under the key `abundances_MEF`.
Finally, the pipeline will return the batch context, which contains the abundances of each protein in each observation in arbitrary units, 
and in MEFs, + any other product of the pipeline application.


