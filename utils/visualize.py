import matplotlib.pyplot as plt
from PIL import Image

def plot_train_validation_loss(train_losses, val_losses):

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

def display_labeled_images(model_name, image_paths, true_labels, predicted_labels):
    # Number of images
    num_images = len(image_paths)

    # Sorting the images based on their filenames
    sorted_indices = sorted(range(len(image_paths)), key=lambda k: int(image_paths[k].stem))
    image_paths = [image_paths[i] for i in sorted_indices]
    true_labels = [true_labels[i] for i in sorted_indices]
    predicted_labels = [predicted_labels[i] for i in sorted_indices]

    # Setting up the figure and axes
    fig, axs = plt.subplots(num_images, 1, figsize=(10, 6 * num_images))

    for i, img_path in enumerate(image_paths):
        # Open and resize the image
        img = Image.open(img_path)
        # We're reducing the image size to a smaller width for easier visualization
        base_width = 300
        w_percent = base_width / float(img.size[0])
        h_size = int(float(img.size[1]) * float(w_percent))
        img = img.resize((base_width, h_size), Image.ANTIALIAS)

        # Display image
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(
            f"Filename: {img_path.name}, True Label: {true_labels[i]}, Predicted Label: {predicted_labels[i]}")

    # Adjusting the space between plots for better readability
    plt.subplots_adjust(hspace=0.2)
    plt.savefig(f'data/labeled_images_{model_name}.png', bbox_inches='tight')
    plt.show()
