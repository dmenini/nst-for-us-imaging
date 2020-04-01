from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from nst_lib import *
from img_lib import *

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

REGULARIZATION = 1


def main():
    # ==================================================================================================================
    # Load and plot input images (content and style)
    # ==================================================================================================================

    content_path = './img/YellowLabradorLooking_new.jpg'
    style_path = './img/Vassily_Kandinsky,_1913_-_Composition_7.jpg'

    content_image = load_img(content_path)
    style_image = load_img(style_path)

    plt.subplot(1, 2, 1)
    imgshow(content_image, 'Content Image')
    plt.subplot(1, 2, 2)
    imgshow(style_image, 'Style Image')
    plt.show()


    # ==================================================================================================================
    # Run NST using pretrained VGG19 model (saved_model)
    # ==================================================================================================================

    nst_module = tf.saved_model.load("./nst_model")
    stylized_image = nst_module(tf.constant(content_image), tf.constant(style_image))[0]
    tensor_to_image(stylized_image)
    imgshow(stylized_image, 'Stylized Image')
    plt.show()

    # ==================================================================================================================
    # Load VGG19 and verify classification accuracy
    # ==================================================================================================================

    x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    prediction_probabilities = vgg(x)
    print(prediction_probabilities.shape)
    predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
    print([(class_name, prob) for (number, class_name, prob) in predicted_top_5])

    # ==================================================================================================================
    # Extract content and style layers from VGG19's intermediate layers
    # ==================================================================================================================

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    print()
    for layer in vgg.layers:
        print(layer.name)

    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2']

    # Style layer of interest
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    # Create a model (input = vgg's input, outputs = vgg's intermediate style outputs)
    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image * 255)

    # Look at the statistics of each layer's output
    for name, output in zip(style_layers, style_outputs):
        print(name)
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())
        print()

    # ==================================================================================================================
    # Create a model that extracts both content and style
    # ==================================================================================================================

    # This model returns a dict of the gram matrix (style) of the style_layers and content of the content_layers
    extractor = StyleContentModel(style_layers, content_layers)
    results = extractor(tf.constant(content_image))

    style_results = results['style']

    print('Styles:')
    for name, output in sorted(results['style'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())
        print()

    print("Contents:")
    for name, output in sorted(results['content'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())

    # ==================================================================================================================
    # Run gradient descent (with regularization term in the loss function)
    # ==================================================================================================================

    # Set style and content target values
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    # Define a tf.Variable to contain the image to optimize
    image = tf.Variable(content_image)

    # The paper recommends LBFGS, but Adam works okay too
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    def style_content_loss(outputs, style_weight=1e-2, content_weight=1e4):
        """Weighted combination of style and content loss"""
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(image):
        total_variation_weight = 30
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            if REGULARIZATION == 1:
                loss += total_variation_weight * total_variation_loss(image)    # tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    start = time.time()

    epochs = 10
    steps_per_epoch = 50

    step = 0
    for n in range(epochs):
        print("Epoch: {}".format(n))
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(".", end='')
        tensor_to_image(image)
        imgshow(image, 'Stylized Image')
        #plt.figure()
        #plt.pause(0.01)
        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end - start))
    tensor_to_image(image)
    imgshow(image, 'Stylized Image')
    plt.show()

    file_name = 'stylized_image.png'
    tensor_to_image(image).save(file_name)


if __name__ == "__main__":
    main()
