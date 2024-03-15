library(brulee)
library(tidymodels)
library(doFuture)
library(tictoc)

# Create a recipe
iris_recipe <- 
  recipe(Species ~.  , data = iris) %>%
  # Add normalization step for numeric variables
  step_normalize(all_numeric_predictors())

# Create a parsnip model using brulee
iris_spec <-
  mlp(hidden_units = tune(),
      penalty = tune(),
      epochs = tune()) |>
  set_engine("brulee") |> 
  set_mode("classification") 

iris_spec

# Create a workflow
iris_wf <- workflow() |> 
  add_recipe(iris_recipe) |> 
  add_model(iris_spec)

iris_wf

# Create a grid for hyperparameter tuning
param_grid <- grid_regular(hidden_units(range = c(10, 100)), epochs(range = c(100,1200)), penalty(range = c(-5, 0)), levels = 10)

# Create a cross-validation object
rset <- vfold_cv(iris, strata = Species, repeats = 2)

# Create a parallel backend
registerDoFuture()
cores<-parallelly::availableCores()
plan(multisession, workers = cores)
plan()


tictoc::tic()
iris_res <- tune_grid(iris_wf, resamples = rset, grid = param_grid)
tictoc::toc()
