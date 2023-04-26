#Attack 1 - uses 1% pixel budget achieved 71% on my computer

library(tidyverse)
library(keras)


setwd("IE332P2")
model<-load_model_tf("./dandelion_model")

# Implementing a simple attack algorithm for fpgm
fpgm_attack <- function(image, epsilon, pixel_budget) {
  # Extract the pixel values from the image
  x <- image_to_array(image)
  
  # Calculate the pixel budget based on the total number of pixels in the image
  pb <- floor(prod(dim(x)) * pixel_budget)
  
  # Sort the pixel values by absolute value
  sx <- sort(abs(x), decreasing = TRUE)
  
  # Select the top pixels based on the pixel budget
  selected_pixels <- sx[1:pb]
  
  # Apply the fpgm attack to the selected pixels
  x[abs(x) %in% selected_pixels] <- epsilon * sign(x[abs(x) %in% selected_pixels]) * sqrt(abs(x[abs(x) %in% selected_pixels]))
  
  # Clip the pixel values to the valid [0, 1] range
  x <- pmax(pmin(x, 1), 0)
  
  # Reshape the array to match the input shape of the model
  x <- array_reshape(x, c(1, dim(x)))
  
  # Scale the pixel values to the range [0, 1]
  x <- x / 255
  
  return(x)
}

# Apply the fpgm attack on the test images
f = list.files("./grass")
target_size = c(224, 224)
accuracy_var <- 0
loss_var <- 0
user_epsilon <- 2
pixel_budget <- 0.01

for (i in f) {
  test_image <- image_load(paste("./grass/", i, sep = ""), target_size = target_size)
  
  # Implementing the fpgm attack
  print(paste("Image: ", which(f == i), " / ", length(f)))
  print("Dimensions of unattacked image: ")
  print(dim(test_image))
  x <- fpgm_attack(test_image, user_epsilon, pixel_budget)
  print("Dimensions of attacked image: ")
  print(dim(x))
  
  # Make a prediction on the adversarial example
  pred <- model %>% predict(x)
  
  if (pred[1, 2] < 0.50) {
    print(i)
  }
  
  accuracy_var <- accuracy_var + pred[1, 2]
  loss_var <- loss_var + pred[1, 1]
}
cat("Attacked Grass Accuracy:", accuracy_var, "Loss:", loss_var, "\n")

#Test against unattacked set
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

#Attack 2 - Gaussian Noise
library(tidyverse)
library(keras)

# Load the model
model <- load_model_tf("./dandelion_model")

# Function to apply the Gaussian noise attack
gaussian_attack <- function(image, epsilon) {
  # Extract the pixel values from the image
  x <- image_to_array(image)
  
  # Calculate the total number of pixels
  num_pixels <- length(x)
  
  # Calculate the pixel budget
  pixel_budget <- floor(num_pixels * 0.01)
  
  # Apply the Gaussian noise attack
  noise <- array(rnorm(num_pixels, sd = epsilon), dim = dim(x))
  noise[order(abs(noise), decreasing = TRUE)[(pixel_budget+1):num_pixels]] <- 0
  x <- x + noise
  
  # Clip the pixel values to the valid [0, 1] range
  x <- pmax(pmin(x, 1), 0)
  
  # Reshape the array to match the input shape of the model
  x <- array_reshape(x, c(1, dim(x)))
  
  # Scale the pixel values to the range [0, 1]
  x <- x / 255
  
  return(x)
}

# Testing block
f <- list.files("./grass")
target_size <- c(224, 224)
accuracy_var <- 0
loss_var <- 0
user_epsilon <- 0.01

for (i in f) {
  test_image <- image_load(paste("./grass/", i, sep = ""), target_size = target_size)
  
  # Apply the Gaussian noise attack
  print(paste("Image: ", which(f == i), " / ", length(f)))
  print("Dimensions of unattacked image: ")
  print(dim(test_image))
  x <- gaussian_attack(test_image, user_epsilon)
  print("Dimensions of attacked image: ")
  print(dim(x))
  
  # Make a prediction on the adversarial example
  pred <- model %>% predict(x)
  
  if (pred[1, 2] < 0.50) {
    print(i)
  }
  
  accuracy_var <- accuracy_var + pred[1, 2]
  loss_var <- loss_var + pred[1, 1]
}
cat("Attacked Grass Accuracy:", accuracy_var, "Loss:", loss_var, "\n")


