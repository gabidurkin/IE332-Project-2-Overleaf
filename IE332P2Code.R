library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)
library(GA)
library(pso)
library(GenSA)
library(gradDescent)


setwd("IE332P2")
model<-load_model_tf("./dandelion_model")

#YOUR ALGORITHM HERE==========================

initialize_population <- function(image, pop_size) {
  num_pixels <- nrow(image) * ncol(image)
  population <- matrix(runif(num_pixels * pop_size), ncol = num_pixels)
  population <- round(population)
  return(population)
}
evaluate_population <- function(image, population) {
  # Calculate the objective function for each individual in the population
  objectives <- apply(population, 1, function(individual) {
    modified_image <- apply_adversarial_changes(image, individual)
    x <- image_to_array(modified_image)
    x <- array_reshape(x, c(1, dim(x)))
    x <- x / 255
    pred <- model %>% predict(x)
    objective <- pred[1, 1] # Assuming class 1 is the target class to fool the classifier
    return(objective)
  })
  return(objectives)
}
select_parents <- function(population, fitness_scores) {
  # Perform tournament selection with a tournament size of 2
  num_parents <- nrow(population) %/% 2
  parents <- matrix(nrow = num_parents, ncol = ncol(population))
  for (i in 1:num_parents) {
    tournament <- sample(nrow(population), size = 2, replace = FALSE)
    if (fitness_scores[tournament[1]] > fitness_scores[tournament[2]]) {
      parents[i,] <- population[tournament[1],]
    } else {
      parents[i,] <- population[tournament[2],]
    }
  }
  return(parents)
}
crossover <- function(parents, crossover_rate) {
  num_offspring <- nrow(parents) * 2
  offspring <- matrix(nrow = num_offspring, ncol = ncol(parents))
  for (i in 1:num_offspring) {
    if (runif(1) < crossover_rate) {
      # Perform uniform crossover
      mask <- sample(c(0, 1), size = ncol(parents), replace = TRUE)
      offspring[i,] <- parents[2*(i %/% 2) + 1,] * mask + parents[2*(i %/% 2) + 2,] * (1 - mask)
    } else {
      # Copy over one of the parents
      offspring[i,] <- parents[i %/% 2,]
    }
  }
  return(offspring)
}
mutate <- function(offspring, mutation_rate) {
  num_mutations <- round(mutation_rate * nrow(offspring) * ncol(offspring))
  mutation_indices <- sample(nrow(offspring) * ncol(offspring), size = num_mutations, replace = FALSE)
  offspring[mutation_indices] <- 1 - offspring[mutation_indices]
  return(offspring)
}
get_best_individual <- function(population) {
  objectives <- evaluate_population(image, population)
  best_index <- which.max(objectives)
  best_individual <- population[best_index,]
  return(best_individual)
}
apply_adversarial_changes <- function(image, changes) {
  modified_image <- image
  
  # Apply adversarial changes to the image
  for (i in seq_along(changes)) {
    # Extract the pixel indices and new pixel values from the changes vector
    pixel_idx <- changes[[i]][1]
    new_pixel_val <- changes[[i]][2]
    
    # Calculate the row and column indices of the pixel
    row_idx <- (pixel_idx - 1) %/% dim(modified_image)[1] + 1
    col_idx <- (pixel_idx - 1) %% dim(modified_image)[1] + 1
    
    # Set the new pixel value in the modified image
    modified_image[row_idx, col_idx, ] <- new_pixel_val
  }
  
  return(modified_image)
}
neighbor_function <- function(solution, epsilon = 0.1) {
  # Generate a neighbor solution by adding a small perturbation
  # to a randomly selected pixel
  neighbor <- list(solution)
  num_pixels <- length(solution)
  pixel_idx <- sample(num_pixels, size = 1)
  perturbation <- runif(1, -epsilon, epsilon)
  new_pixel_val <- solution[[pixel_idx]][2] + perturbation
  new_pixel_val <- pmax(pmin(new_pixel_val, 1), 0) # Clip pixel value to [0, 1] range
  neighbor[[pixel_idx]][2] <- new_pixel_val
  
  return(neighbor)
}
select_pixels <- function(image, pixel_budget) {
  # Select pixels to change based on their contribution to the objective function
  # and the pixel budget
  num_pixels <- dim(image)[1] * dim(image)[2]
  x <- image_to_array(image)
  x <- array_reshape(x, c(num_pixels, dim(image)[3]))
  x <- x / 255
  preds <- model %>% predict(x)
  obj_values <- preds[, 1]
  obj_gradient <- k_gradients(loss = k_variable(obj_values), xs = k_constant(x))[[1]]
  obj_gradient_mags <- sqrt(rowSums(obj_gradient ^ 2))
  pixel_order <- order(obj_gradient_mags, decreasing = TRUE)
  
  # Choose the pixels to change based on their contribution to the objective function
  # and the pixel budget
  selected_pixels <- list()
  selected_pixel_count <- 0
  for (i in pixel_order) {
    if (selected_pixel_count < pixel_budget) {
      row_idx <- (i - 1) %/% dim(image)[1] + 1
      col_idx <- (i - 1) %% dim(image)[1] + 1
      new_pixel_val <- image[row_idx, col_idx, ] + obj_gradient[i, ]
      new_pixel_val <- pmax(pmin(new_pixel_val, 255), 0) # Clip pixel value to [0, 255] range
      selected_pixels[[i]] <- c(i, new_pixel_val)
      selected_pixel_count <- selected_pixel_count + 1
    } else {
      break
    }
  }
  
  return(selected_pixels)
}





setwd("IE332P2")
model <- load_model_tf("./dandelion_model")

# Placeholder functions for the individual algorithms
genetic_algorithm <- function(image) {
  # Parameters for the Genetic Algorithm
  pop_size <- 50
  num_generations <- 100
  mutation_rate <- 0.1
  crossover_rate <- 0.8
  
  # Initialize population
  population <- initialize_population(image, pop_size)
  
  # Iterate through generations
  for (generation in 1:num_generations) {
    # Evaluate the fitness of the population
    fitness_scores <- evaluate_population(image, population)
    
    # Select parents based on fitness scores
    parents <- select_parents(population, fitness_scores)
    
    # Perform crossover to generate offspring
    offspring <- crossover(parents, crossover_rate)
    
    # Apply mutation to offspring
    offspring <- mutate(offspring, mutation_rate)
    
    # Replace the population with offspring
    population <- offspring
  }
  
  # Choose the best individual from the final population
  best_individual <- get_best_individual(population)
  
  # Apply the adversarial changes based on the best individual
  modified_image <- apply_adversarial_changes(image, best_individual)
  
  return(modified_image)
}

particle_swarm_optimization <- function(image) {
  # Parameters for Particle Swarm Optimization
  swarm_size <- 30
  num_iterations <- 100
  w <- 0.7 # Inertia weight
  c1 <- 2  # Cognitive component weight
  c2 <- 2  # Social component weight
  
  # Set up fitness function
  fitness_function <- function(particles) {
    # Calculate fitness for each particle
    fitness_scores <- sapply(particles, function(p) evaluate_particle(image, p))
    return(fitness_scores)
  }
  
  # Initialize swarm
  swarm <- initialize_swarm(image, swarm_size)
  
  # Run Particle Swarm Optimization
  pso_result <- pso::psoptim(swarm_size, fitness_function, num_iterations, w, c1, c2)
  
  # Get the best particle
  best_particle <- pso_result$gbest
  
  # Apply the adversarial changes based on the best particle
  modified_image <- apply_adversarial_changes(image, best_particle)
  
  return(modified_image)
}

simulated_annealing <- function(image) {
  # Define the objective function
  objective_function <- function(p) {
    modified_image <- apply_adversarial_changes(image, p)
    x <- image_to_array(modified_image)
    x <- array_reshape(x, c(1, dim(x)))
    x <- x / 255
    pred <- model %>% predict(x)
    
    # Calculate objective based on classifier output
    objective <- pred[1, 1] # Assuming class 1 is the target class to fool the classifier
    return(objective)
  }
  
  # Initialize solution
  initial_solution <- initialize_solution(image)
  
  # Set the lower and upper bounds for the search space
  lower_bounds <- rep(0, length(initial_solution))
  upper_bounds <- rep(1, length(initial_solution))
  
  # Perform Simulated Annealing
  result <- GenSA(
    fn = objective_function,
    lower = lower_bounds,
    upper = upper_bounds,
    init = initial_solution
  )
  
  # Get the best solution found by Simulated Annealing
  best_solution <- result$par
  
  # Apply the adversarial changes based on the best solution
  modified_image <- apply_adversarial_changes(image, best_solution)
  
  return(modified_image)
}

FGSM_function <- function(image, epsilon) {
  # Convert the input image to an array and reshape it
  x <- image_to_array(image)
  x <- array_reshape(x, c(1, dim(x)))
  x <- x / 255
  
  # Create a Keras variable from the input image array
  input_image <- k_variable(x)
  
  # Get the model's predictions
  preds <- model(input_image)
  
  # Get the index of the class with the highest predicted probability
  top_class_idx <- k_argmax(preds, axis = -1)
  
  # Define the loss function
  loss <- k_sparse_categorical_crossentropy(y_true = top_class_idx, y_pred = preds)
  
  # Get the gradients of the loss function with respect to the input image
  grads <- k_gradients(loss, input_image)[[1]]
  
  # Compute the sign of the gradients
  signed_grads <- k_sign(grads)
  
  # Apply the FGSM perturbation
  perturbed_image <- input_image + epsilon * signed_grads
  
  # Clip the perturbed image to be in the valid pixel range [0, 1]
  clipped_perturbed_image <- k_clip(perturbed_image, 0, 1)
  
  # Convert the perturbed image back to a regular R array
  modified_image <- k_eval(clipped_perturbed_image)
  
  # Rescale pixel values back to the original range [0, 255]
  modified_image <- modified_image * 255
  
  return(modified_image)
}

hill_climbing <- function(image) {
  # Define the objective function
  objective_function <- function(p) {
    modified_image <- apply_adversarial_changes(image, p)
    x <- image_to_array(modified_image)
    x <- array_reshape(x, c(1, dim(x)))
    x <- x / 255
    pred <- model %>% predict(x)
    
    # Calculate objective based on classifier output
    objective <- pred[1, 1] # Assuming class 1 is the target class to fool the classifier
    return(objective)
  }
  
  # Initialize solution
  current_solution <- initialize_solution(image)
  
  # Set number of iterations
  num_iterations <- 1000
  
  # Hill Climbing loop
  for (i in 1:num_iterations) {
    # Generate a neighbor
    neighbor <- neighbor_function(current_solution)
    
    # Evaluate the objective function for the neighbor
    neighbor_objective <- objective_function(neighbor)
    
    # Compare neighbor's objective with the current solution's objective
    if (neighbor_objective > objective_function(current_solution)) {
      # If the neighbor has a better objective, update the current solution
      current_solution <- neighbor
    }
  }
  
  # Apply the adversarial changes based on the best solution
  modified_image <- apply_adversarial_changes(image, current_solution)
  
  return(modified_image)
}


# Train the individual algorithms on the provided training data
# ...

# Optimize the weights for the individual algorithms
# ...

# Implement the main_algorithm function
main_algorithm <- function(image, pixel_budget) {
  result_ga <- genetic_algorithm(image)
  result_pso <- particle_swarm_optimization(image)
  result_sa <- simulated_annealing(image)
  result_gd <- FGSM_function(image, 0.01)
  result_rf <- hill_climbing(image)
  
  # Assign optimized weights to the results of the algorithms
  weights <- c(weight_ga, weight_pso, weight_sa, weight_gd, weight_rf)
  
  # Combine the results using the weighted majority classifier
  weighted_results <- result_ga * weights[1] +
    result_pso * weights[2] +
    result_sa * weights[3] +
    result_gd * weights[4] +
    result_rf * weights[5]
  
  # Determine the final output by selecting the pixels
  # to change based on the weighted results and the pixel budget
  final_output <- select_pixels(weighted_results, pixel_budget)
  
  return(final_output)
}

# Apply the adversarial attack on the test images using the main_algorithm function
# ...



#TESTING==========================
#The images are already classified into the appropriate folders, and you can use
#the following code after you've modified your images to determine if you made the
#classifier fail (very similar to what's in the tutorial, only slightly modified
#to let you check all the images in the grass or dandelion folders, respectively)

res=c("","")
f=list.files("./grass")
target_size = c(224, 224)
for (i in f){
  test_image <- image_load(paste("./grass/",i,sep=""),
                           target_size = target_size)
  x <- image_to_array(test_image)
  x <- array_reshape(x, c(1, dim(x)))
  x <- x/255
  pred <- model %>% predict(x)
  if(pred[1,2]<0.50){
    print(i)
  }
}

res=c("","")
f=list.files("./dandelions")
for (i in f){
  test_image <- image_load(paste("./dandelions/",i,sep=""),
                           target_size = target_size)
  x <- image_to_array(test_image)
  x <- array_reshape(x, c(1, dim(x)))
  x <- x/255
  pred <- model %>% predict(x)
  if(pred[1,1]<0.50){
    print(i)
  }
}
print(res)
