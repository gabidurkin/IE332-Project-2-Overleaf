# Load required libraries
library(keras)
library(tensorflow)

model <- load_model_tf("./dandelion_model")
target_size <- c(224, 224)

#Set to 1 for grass or 2 for dandelions
target_class <- 1

if (target_class == 2) {
  f <- list.files("./dandelions")
  filepaththing <- "./dandelions/"
} else {
  f <- list.files("./grass")
  filepaththing <- "./grass/"
}


for (i in f) {
  initial_image_image <- image_load(paste(filepaththing, i, sep=""), target_size = target_size)
  initial_image <- image_to_array(initial_image_image)
  
  # Preprocess the image
  input_image <- array_reshape(initial_image / 255, c(1, target_size[[1]], target_size[[2]], 3))
  input_image_tensor <- tf$constant(input_image, dtype = tf$float32)
  
  # Function to calculate the saliency map
  calculate_saliency <- function(model, input_image) {
    with(tf$GradientTape() %as% tape, {
      tape$watch(input_image)
      output <- model(input_image)
    })
    gradients <- tape$gradient(output, input_image)
    squared_gradients <- tf$square(gradients)
    saliency_map <- tf$reduce_sum(squared_gradients, axis = -1L)
    return(saliency_map)
  }
  
  # Calculate the saliency map
  saliency_map_tensor <- calculate_saliency(model, input_image_tensor)
  saliency_map <- as.array(saliency_map_tensor)[1, ,]
  
  # Normalize the saliency map to get a relative importance rating for each pixel
  saliency_map_normalized <- saliency_map / (max(saliency_map) + 1e-10)
  
  # Find the top 1% of pixels with the highest saliency values
  num_pixels_to_manipulate <- ceiling(0.01 * prod(dim(saliency_map_normalized)))
  saliency_map_flat <- as.vector(saliency_map_normalized)
  sorted_indices <- order(saliency_map_flat, decreasing = TRUE)
  top_pixel_indices <- sorted_indices[1:num_pixels_to_manipulate]
  
  # Convert the selected indices back to their corresponding 3D positions
  top_pixel_positions <- arrayInd(top_pixel_indices, dim(saliency_map_normalized))
  
  # Create a copy of the input image to modify
  modified_image <- input_image
  
  # Loop through the top pixel positions and invert the pixel values
  for (idx in seq_along(top_pixel_positions[, 1])) {
    x <- top_pixel_positions[idx, 1]
    y <- top_pixel_positions[idx, 2]
    
    for (channel in 1:3) {
      modified_image[1, x, y, channel] <- 255 - input_image[1, x, y, channel] # Invert the pixel value (scaled between 0 and 255)
    }
    modified_image <- array_reshape(modified_image / 255, c(1, target_size[[1]], target_size[[2]], 3))
  }
  
  # The saliency_map_normalized is a 2D array with the same height and width as the input image
  # Each value represents the relative importance of changing the corresponding pixel in the input image
  if (sum(saliency_map_normalized)==0) {
    cat("\n❌ Can't get gradients on model I guess?: "," ",sum(saliency_map_normalized)," ",i)
  }
  else {
    cat("\n✅ "," ",sum(saliency_map_normalized)," ",i)
  }
  OGaccloss <- as.array(model(input_image))
  Attackedaccloss <- as.array(model(modified_image))
  cat("\nLoss: ",OGaccloss[target_class]," -> ",Attackedaccloss[target_class],"\n")

}

