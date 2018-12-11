import numpy as np
from functions.io_data import read_data, write_data

np.random.seed(0)

class IsingModel():

    def __init__(self, image, J, ex_factor=None):
        self.image = image
        self.width = image.shape[0]
        self.height = image.shape[1]
        self.external_filed = None
        if ex_factor is not None:
            self.external_filed = ex_factor * image
        self._J = J

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

        if self.external_filed is not None:
            return self.external_filed[x, y] + sum(nbrs)
        else:
            return sum(nbrs)

    def gibbs_sampling(self, x, y):
        p = 1 / (1 + np.exp(-2 * self._J * self.interaction_potentials(x, y)))

        if p > np.random.uniform():
            self.image[x, y] = 1
        else:
            self.image[x, y] = -1

def denoising(image, burn_in, iterations, q=None, threshold=0.7, J=3):
    external_factor = None
    if q is not None:
        external_factor = 0.5 * np.log(q / (1 - q))

    ising = IsingModel(image, J=J, ex_factor=external_factor)

    average = np.zeros(image.shape).astype(np.float64)

    for i in range(burn_in + iterations):
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if np.random.uniform() < threshold:
                    ising.gibbs_sampling(x, y)

        if i > burn_in:
            average += ising.image

    denoised = average / iterations

    return denoised

if __name__ == "__main__":
    for img in range(1,5):
        print("Denoising for image " + str(img))
        data, image = read_data("../a1/"+str(img)+"_noise.txt", True)

        print(data.shape)
        print(image.shape)

        image[image == 0] = -1
        image[image == 255] = 1

        burn_in = 5
        iterations = 10
        q = 0.7 # 0.x or None # for logit - external filed factor
        threshold = 0.7

        d_img = denoising(image, burn_in=burn_in, iterations=iterations, q=q, threshold=threshold)

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

        write_data(data, "../output/gibbs/"+str(img)+"_denoise.txt")
        read_data("../output/gibbs/"+str(img)+"_denoise.txt", True, save=True, save_name="../output/gibbs/"+str(img)+"_denoise.png")
        print("Finished writing data. Please check "+str(img)+"_denoise.png \n")
