import numpy as np
from scipy.stats import multivariate_normal
from functions.io_data import read_data, write_data

np.random.seed(0)

class IsingModel():

    def __init__(self, image, J, rate, sigma):
        self.width = image.shape[0]
        self.height = image.shape[1]
        self._J = J
        self._rate = rate
        self._sigma = sigma

        self.image, self.logodds = self.presenting_image(image)

    def presenting_image(self, image):
        logodds = multivariate_normal.logpdf(image.flatten(), mean=+1, cov=self._sigma ** 2) - multivariate_normal.logpdf(image.flatten(), mean=-1, cov=self._sigma ** 2)
        logodds = np.reshape(logodds, image.shape)
        pr_plus1 = 1 / (1 + np.exp(-1*logodds))  # sigmoid(logodds)  # plus 1 -> +1 -> 1 / (1 + exp(logodds)) -> sigmoid(x) = 1 / (1 + exp{x})
        return 2 * pr_plus1 - 1, logodds

    def neighbors(self, x, y):
        nbrs = []

        if x == 0:
            nbrs.append(self.image[self.width - 1, y])
        else:
            nbrs.append(self.image[x - 1, y])

        if x == self.width - 1:
            nbrs.append(self.image[0, y])
        else:
            nbrs.append(self.image[x + 1, y])

        if y == 0:
            nbrs.append(self.image[x, self.height - 1])
        else:
            nbrs.append(self.image[x, y - 1])

        if y == self.height - 1:
            nbrs.append(self.image[x, 0])
        else:
            nbrs.append(self.image[x, y + 1])

        return nbrs

    def interaction_potentials(self, x, y):
        nbrs = self.neighbors(x, y)
        return sum(nbrs)

    def variational_inference(self, x, y):
        E = self._J * self.interaction_potentials(x, y)
        self.image[x, y] = (1 - self._rate) * self.image[x, y] + self._rate * np.tanh(E + 0.5 * self.logodds[x, y])


def denoising(image, iterations, rate, sigma, J=3):
    ising = IsingModel(image, J=J, rate=rate, sigma=sigma)

    for i in range(iterations):
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                ising.variational_inference(x, y)

    return ising.image

if __name__ == "__main__":
    for img in range(1,5):
        print("Denoising for image " + str(img))
        data, image = read_data("../a1/"+str(img)+"_noise.txt", True)

        print(data.shape)
        print(image.shape)

        image[image == 0] = -1
        image[image == 255] = 1

        iterations = 15
        J = 3
        sigma = 2
        rate = 0.5

        d_img = denoising(image, iterations=iterations, rate=rate, sigma=sigma)

        d_img[d_img >= 0] = 255
        d_img[d_img < 0] = 0

        print(d_img.shape)
        height = d_img.shape[0]
        width = d_img.shape[1]
        counter = 0

        for i in range(0, width):
            for j in range(0, height):
                data[counter][2] = d_img[j][i][0]
                counter = counter + 1

        write_data(data, "../output/vi/"+str(img)+"_denoise.txt")
        read_data("../output/vi/"+str(img)+"_denoise.txt", True, save=True, save_name="../output/vi/"+str(img)+"_denoise.png")
        print("Finished writing data. Please check "+str(img)+"_denoise.png \n")