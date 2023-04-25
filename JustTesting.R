library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)
library(GA)
library(pso)
library(GenSA)

#setwd("IE332P2")
model<-load_model_tf("./dandelion_model")

#YOUR ALGORITHM HERE==========================

initialize_population <- function(image, pop_size) {
  num_pixels <- nrow(image) * ncol(image) * dim(image)[3] # Multiply by the number of channels (dim(image)[3])
  population <- matrix(runif(num_pixels * pop_size, min = 0, max = 255), ncol = num_pixels)
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
    cat("\n",individual[1],individual[2],individual[3],individual[4],individual[5],individual[11],individual[12],individual[13],individual[14],individual[15],individual[16],individual[101],individual[102],individual[103],individual[104])
    cat("\n",x[1],x[2],x[3],x[4],x[5],x[11],x[12],x[13],x[14],x[15],x[16],x[101],x[102],x[103],x[104])
    pred <- model %>% predict(x)
    print(pred)
    objective <- pred[1, 2] # Assuming class 1 is the target class to fool the classifier
    return(objective)
  })
  return(objectives)
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
    
    # Calculate the row, column, and channel indices of the pixel
    num_rows <- dim(modified_image)[1]
    num_cols <- dim(modified_image)[2]
    row_idx <- (pixel_idx - 1) %/% (num_rows * num_cols) %% num_rows + 1
    col_idx <- (pixel_idx - 1) %/% num_rows %% num_cols + 1
    channel_idx <- (pixel_idx - 1) %/% (num_rows * num_cols * 3) + 1
    
    # Set the new pixel value in the modified image
    modified_image[row_idx, col_idx, channel_idx] <- new_pixel_val
  }
  
  return(modified_image)
}


particle_swarm_optimization <- function(image) {
  # Parameters for Particle Swarm Optimization
  num_particles <- 2
  num_iterations <- 2
  w <- 2.7
  c1 <- 10.5
  c2 <- 10.5
  
  # Initialize particles
  particles <- initialize_population(image, num_particles)
  cat("dim(particles): ", dim(particles))
  # Initialize particle velocities
  velocities <- matrix(0, nrow = num_particles, ncol = ncol(particles))
  
  # Initialize best individual positions and global best position
  p_best_positions <- particles
  p_best_scores <- rep(0, num_particles)
  g_best_position <- particles[1,]
  g_best_score <- 0
  
  # Iterate through iterations
  for (iteration in 1:num_iterations) {
    # Evaluate the fitness of the particles
    cat("\nIteration: ", iteration, " / ", num_iterations, "\n")
    fitness_scores <- -evaluate_population(image, particles)
    
    # Update the best individual positions and scores
    better_scores_idx <- fitness_scores > p_best_scores
    p_best_positions[better_scores_idx,] <- particles[better_scores_idx,]
    p_best_scores[better_scores_idx] <- fitness_scores[better_scores_idx]
    
    # Update the global best position and score
    if (max(fitness_scores) > g_best_score) {
      g_best_position <- particles[which.max(fitness_scores),]
      g_best_score <- max(fitness_scores)
    }
    
    # Update particle velocities
    r1 <- runif(ncol(particles))
    r2 <- runif(ncol(particles))
    cognitive_velocity <- c1 * r1 * (p_best_positions - particles)
    social_velocity <- c2 * r2 * (matrix(rep(g_best_position, num_particles), nrow = num_particles, byrow = TRUE) - particles)
    velocities <- w * velocities + cognitive_velocity + social_velocity
    
    # Update particle positions
    particles <- particles + velocities
    particles <- pmax(pmin(particles, 255), 0) # Clip particle values to [0, 255] range
    
  }
  
  # Apply the adversarial changes based on the global best position
  modified_image <- apply_adversarial_changes(image, g_best_position)
  print(g_best_position)
  return(modified_image)
}



# Apply the adversarial attack on the test images
f=list.files("./grass")
target_size = c(224, 224)
accuracy_var <- 0
loss_var <- 0
for (i in f){
  test_image <- image_load(paste("./grass/",i,sep=""), target_size = target_size)
  x <- image_to_array(test_image)
  
  #Implementing adversarial attack with just one algorithm
  print(paste("Image: ", which(f == i), " / ", length(f)))
  print("Dimensions of unattacked image: ")
  print(dim(x))
  x <- particle_swarm_optimization(x)
  print("Dimensions of attacked image: ")
  print(dim(x))
  x <- array_reshape(x, c(1, dim(x)))
  x <- x/255
  pred <- model %>% predict(x)
  if(pred[1,2]<0.50){
    print(i)
  }
  accuracy_var <- accuracy_var + pred[1,2]
  loss_var <- loss_var + pred[1,1]
}
cat("Attacked Grass Accuracy:", accuracy_var, "Loss:", loss_var, "\n")

#TESTING==========================
#The images are already classified into the appropriate folders, and you can use
#the following code after you've modified your images to determine if you made the
#classifier fail (very similar to what's in the tutorial, only slightly modified
#to let you check all the images in the grass or dandelion folders, respectively)

f=list.files("./grass")
target_size = c(224, 224)
accuracy_var <- 0
loss_var <- 0
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
  accuracy_var <- accuracy_var + pred[1,2]
  loss_var <- loss_var + pred[1,1]
}
cat("Unchanged Grass Accuracy:", accuracy_var, "Loss:", loss_var, "\n")



#DECLAN CODE

library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)
library(GA)
library(pso)
library(GenSA)

setwd("IE332P2")
model<-load_model_tf("./dandelion_model")

#YOUR ALGORITHM HERE
initialize_population <- function(image, pop_size) {
  num_pixels <- nrow(image) * ncol(image) * dim(image)[3] # Multiply by the number of channels (dim(image)[3])
  population <- matrix(runif(num_pixels * pop_size, min = 0, max = 255), ncol = num_pixels)
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
    cat("\n",individual[1],individual[2],individual[3],individual[4],individual[5],individual[11],individual[12],individual[13],individual[14],individual[15],individual[16],individual[101],individual[102],individual[103],individual[104])
    cat("\n",x[1],x[2],x[3],x[4],x[5],x[11],x[12],x[13],x[14],x[15],x[16],x[101],x[102],x[103],x[104])
    pred <- model %>% predict(x)
    print(pred)
    objective <- pred[1, 2] # Assuming class 1 is the target class to fool the classifier
    return(objective)
  })
  return(objectives)
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
    
    # Calculate the row, column, and channel indices of the pixel
    num_rows <- dim(modified_image)[1]
    num_cols <- dim(modified_image)[2]
    row_idx <- (pixel_idx - 1) %/% (num_rows * num_cols) %% num_rows + 1
    col_idx <- (pixel_idx - 1) %/% num_rows %% num_cols + 1
    channel_idx <- (pixel_idx - 1) %/% (num_rows * num_cols * 3) + 1
    
    # Set the new pixel value in the modified image
    modified_image[row_idx, col_idx, channel_idx] <- new_pixel_val
  }
  
  return(modified_image)
}

adversarial_attack_gaussian <- function(image, noise_sd) {
  # Generate Gaussian noise with standard deviation noise_sd
  noise <- array(rnorm(nrow(image) * ncol(image) * dim(image)[3], mean = 0, sd = noise_sd), dim = dim(image))
  noisy_image <- image + noise
  
  # Clip the pixel values to [0, 255]
  noisy_image <- pmax(noisy_image, 0)
  noisy_image <- pmin(noisy_image, 255)
  
  # Convert the image to a format that can be fed to the model
  x <- image_to_array(noisy_image)
  x <- array_reshape(x, c(1, dim(x)))
  x <- x / 255
  
  # Use the model to make a prediction on the noisy image
  pred <- model %>% predict(x)
  
  # Return the noisy image and the predicted class probabilities
  return(list(image = noisy_image, pred = pred))
}

# Apply the adversarial attack on the test images
f=list.files("./grass")
target_size = c(224, 224)
accuracy_var <- 0
loss_var <- 0
sd <- 0.1
for (i in f){
  test_image <- image_load(paste("./grass/",i,sep=""), target_size = target_size)
  x <- image_to_array(test_image)
  
  #Implementing adversarial attack with just one algorithm
  #print(paste("Image: ", which(f == i), " / ", length(f)))
  #print("Dimensions of unattacked image: ")
  #print(dim(x))
  x <- adversarial_attack_gaussian(x, sd)
  x <- x$image  # <--- Add this line to extract the modified image from the returned list
  #print("Dimensions of attacked image: ")
  #print(dim(x))
  x <- array_reshape(x, c(1, dim(x)))
  x <- x/255
  pred <- model %>% predict(x)
  if(pred[1,2]<0.50){
    print(i)
  }
  accuracy_var <- accuracy_var + pred[1,2]
  loss_var <- loss_var + pred[1,1]
}
cat("Attacked Grass Accuracy:", accuracy_var, "Loss:", loss_var, "\n")


