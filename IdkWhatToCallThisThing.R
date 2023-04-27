# Load required libraries
library(keras)
library(tensorflow)

model <- load_model_tf("./dandelion_model")
target_size <- c(224, 224)

#Set to 1 for grass or 2 for dandelions
target_class <- 2
pixel_budget_ratio <- 0.01

epsilon <- 0.1
pixel_split = 30
deps <- 10
extra_reps = 30
pixels_per_split = round(pixel_budget_ratio * target_size[[1]] * target_size[[2]] / pixel_split)

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
  with(tf$GradientTape() %as% tape, {
    tape$watch(input_image_tensor)
    output <- model(input_image_tensor)[1,(target_class)]
  })
  gradients_tensor <- tape$gradient(output, input_image_tensor)
  gradients <- as.array(gradients_tensor)[1, , ,]
  # Calculate the saliency map
  saliency_map_tensor <- tf$reduce_sum(tf$square(gradients_tensor), axis = -1L)
  saliency_map <- as.array(saliency_map_tensor)[1, ,]
  # Find the top 1% / pixel_split of pixels with the highest saliency values
  saliency_map_flat <- as.vector(saliency_map)
  sorted_indices <- order(saliency_map_flat, decreasing = TRUE)
  top_pixel_indices <- sorted_indices[1:pixels_per_split]
  # Convert the selected indices back to their corresponding 3D positions
  top_pixel_positions <- arrayInd(top_pixel_indices, dim(saliency_map_normalized))
  
  # Create a copy of the input image to modify
  modified_image <- input_image

  # Loop through the top pixel positions and invert the pixel values
  for (idx in seq_along(top_pixel_positions[, 1])) {
    x <- top_pixel_positions[idx, 1]
    y <- top_pixel_positions[idx, 2]
    
    for (channel in 1:3) {
      if (gradients[x,y,channel] > 0) {
        modified_image[1, x, y, channel] <- (1 - modified_image[1, x, y, channel])*epsilon + modified_image[1, x, y, channel]
      } else {
        modified_image[1, x, y, channel] <- -(modified_image[1, x, y, channel])*epsilon + modified_image[1, x, y, channel]
      }
    }
  }
  for (reps in 1:(pixel_split+1)) {
    #print(reps)
    modified_image_tensor <- tf$constant(modified_image, dtype = tf$float32)
    with(tf$GradientTape() %as% tape, {
      tape$watch(modified_image_tensor)
      output <- model(modified_image_tensor)[1,(target_class)]
    })
    gradients_tensor <- tape$gradient(output, modified_image_tensor)
    gradients <- as.array(gradients_tensor)[1, , ,]
    
    # Calculate the saliency map
    saliency_map_tensor <- tf$reduce_sum(tf$square(gradients_tensor), axis = -1L)
    saliency_map <- as.array(saliency_map_tensor)[1, ,]
    saliency_map_flat <- as.vector(saliency_map)
    sorted_indices <- order(saliency_map_flat, decreasing = TRUE)
    
    # Initialize a counter for the number of unique pixels added
    num_unique_pixels_added <- 0
    for (indexi in 1:length(sorted_indices)) {
      if (!(sorted_indices[indexi] %in% top_pixel_indices)) {
        top_pixel_indices <- c(top_pixel_indices, sorted_indices[indexi])
        num_unique_pixels_added <- num_unique_pixels_added + 1
      }
      if (num_unique_pixels_added >= pixels_per_split || length(top_pixel_indices) > floor(pixel_budget_ratio*target_size[[1]]*target_size[[2]])) { break }
    }
    top_pixel_positions <- arrayInd(top_pixel_indices, dim(saliency_map))
    
    
    for (idx in seq_along(top_pixel_positions[, 1])) {
      x <- top_pixel_positions[idx, 1]
      y <- top_pixel_positions[idx, 2]
      
      for (channel in 1:3) {
        if (gradients[x,y,channel] > 0) {
          modified_image[1, x, y, channel] <- (1 - modified_image[1, x, y, channel])*(deps*epsilon) + modified_image[1, x, y, channel]
        } else {
          modified_image[1, x, y, channel] <- -(modified_image[1, x, y, channel])*(deps*epsilon) + modified_image[1, x, y, channel]
        }
      }
    }
  }
  for (reps in 1:extra_reps) {
    #print(reps)
    modified_image_tensor <- tf$constant(modified_image, dtype = tf$float32)
    with(tf$GradientTape() %as% tape, {
      tape$watch(modified_image_tensor)
      output <- model(modified_image_tensor)[1,(target_class)]
    })
    gradients_tensor <- tape$gradient(output, modified_image_tensor)
    gradients <- as.array(gradients_tensor)[1, , ,]
    
    for (idx in seq_along(top_pixel_positions[, 1])) {
      x <- top_pixel_positions[idx, 1]
      y <- top_pixel_positions[idx, 2]
      for (channel in 1:3) {
        if (gradients[x,y,channel] > 0) {
          modified_image[1, x, y, channel] <- (1 - modified_image[1, x, y, channel])*epsilon + modified_image[1, x, y, channel]
        } else {
          modified_image[1, x, y, channel] <- -(modified_image[1, x, y, channel])*epsilon + modified_image[1, x, y, channel]
        }
      }
    }
  }

  # The saliency_map_normalized is a 2D array with the same height and width as the input image
  # Each value represents the relative importance of changing the corresponding pixel in the input image
  if (sum(saliency_map_normalized)==0) {
    cat("\n❌ Can't get gradients on model I guess?: "," ",sum(saliency_map_normalized)," ",i)
  }
  else {
    cat("\n✅ Image: ", which(f == i), " / ", length(f),"\n",i)
  }
  OGaccloss <- as.array(model(input_image))
  Attackedaccloss <- as.array(model(modified_image))
  cat("\nLoss: ",OGaccloss[target_class]," -> ",Attackedaccloss[target_class],"\n")

}

