import numpy as np
from scipy.stats import multivariate_normal
# from sklearn.cluster import KMeans
from functions.io_data import read_data, write_data
from functions.KMeans import KMeans

np.random.seed(0)

'''
  pi_k: mixing coefficients
  mu_k: mean of Gaussian Distribution
  cov_k: standard deviation of Gaussian Distribution
'''
class EM():

    def __init__(self, data, image,
                 mean1=None, mean2=None,
                 cov1=None, cov2=None,
                 mix1=0.4, mix2=0.6,
                 filename='keke', threshold=0.5):
        self.height = image.shape[0]
        self.width= image.shape[1]
        self.image = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))

        self._mu1 = [32, -12, 19] # [.0, .0, .0]
        if mean1 is not None:
            self._mu1 = mean1
        self._mu2 = [82, 0, 12] # [.0, .0, .0]
        if mean2 is not None:
            self._mu2 = mean2

        self._cov1 = np.cov([[37, -14, 19], [35, -14, 21], [36, -12, 18]]) #[[.0, .0, .0], [.0, .0, .0], [.0, .0, .0]]
        if cov1 is not None:
            self._cov1 = cov1
        self._cov2 = np.cov([[38, -12, 22], [35, -13, 19], [36, -12, 19]]) #[[.0, .0, .0], [.0, .0, .0], [.0, .0, .0]]
        if cov2 is not None:
            self._cov2 = cov2

        self._pi1 = mix1
        self._pi2 = mix2

        self.threshold = threshold
        self.mles = []

        self._filename = filename
        self.data_template = data

    def execute(self):
        flag = True
        while flag:
            self.upgrading()
            self.evaluate_likelihood()
            flag = self.check_flag()
        self.retrieve_picture()

    def responsibility(self):
        '''

        :return: gammas: in shape (N, 1), where N is the # pixels
        '''
        # x is (l, a, b)
        gamma1 = self._pi1 * multivariate_normal.pdf(self.image, mean=self._mu1, cov=self._cov1, allow_singular=True)
        gamma2 = self._pi2 * multivariate_normal.pdf(self.image, mean=self._mu2, cov=self._cov2, allow_singular=True)

        sum_gamma = gamma1 + gamma2

        return gamma1/sum_gamma, gamma2/sum_gamma

    def upgrading(self):
        gamma1, gamma2 = self.responsibility()

        N1 = np.sum(gamma1)
        N2 = np.sum(gamma2)

        self._mu1 = np.dot(self.image.T, gamma1) / N1
        self._mu2 = np.dot(self.image.T, gamma2) / N2

        self._cov1 = np.dot(gamma1 * (self.image - self._mu1).T, (self.image - self._mu1)) / N1
        self._cov2 = np.dot(gamma2 * (self.image - self._mu2).T, (self.image - self._mu2)) / N2

        self._pi1 = N1 / self.image.shape[0]
        self._pi2 = N2 / self.image.shape[0]

    def evaluate_likelihood(self):
        prob1 = multivariate_normal.pdf(self.image, mean=self._mu1, cov=self._cov1, allow_singular=True)
        prob2 = multivariate_normal.pdf(self.image, mean=self._mu2, cov=self._cov2, allow_singular=True)

        expectation = self._pi1 * prob1 + self._pi2 * prob2
        log_likelihood = np.sum(np.log(expectation))

        self.mles.append(log_likelihood)

        print("Log Likelihood: " + str(log_likelihood))

    def check_flag(self):
        if len(self.mles) < 2: return True
        if np.abs(self.mles[-1] - self.mles[-2]) < self.threshold: return False
        return True

    def retrieve_picture(self):
        mask = self.data_template.copy()
        foreground = self.data_template.copy()
        background = self.data_template.copy()

        for i in range(self.image.shape[0]):
            pixel = self.image[i]

            prob1 = multivariate_normal.pdf(pixel, mean=self._mu1, cov=self._cov1, allow_singular=True)
            prob2 = multivariate_normal.pdf(pixel, mean=self._mu2, cov=self._cov2, allow_singular=True)

            # calculate the probility with bernoulli distribution -> Pr[Z_1|x] = Bern[rho], Pr[Z_2|x] = Bern[1-rho]
            # responsibility1 = self._pi1 * prob1
            # responsibility2 = self._pi2 * prob2
            #
            # responsibility1 = responsibility1 / (responsibility1 + responsibility2)
            # responsibility2 = responsibility2 / (responsibility1 + responsibility2)

            expectation = self._pi1 * prob1 + self._pi2 * prob2
            c1 = prob1 / expectation
            c2 = prob2 / expectation

            y = int(i / self.width) + 1
            x = int(i % self.width)

            j = int(x*self.height + y) - 1

            if (c1 < c2):
                background[j][2] = background[j][3] = background[j][4] = 0
                mask[j][2] = mask[j][3] = mask[j][4] = 0
                # print('background: {} - ({}, {})'.format(j, x, y))
            else:
                foreground[j][2] = foreground[j][3] = foreground[j][4] = 0
                mask[j][2] = 100
                mask[j][3] = mask[j][4] = 0
                # print('foreground: {} - ({}, {})'.format(j, x, y))

        write_data(background, "../output/EM/" + str(self._filename) + "_back.txt")
        read_data("../output/EM/" + str(self._filename) + "_back.txt", False, save=True, save_name="../output/EM/" + str(self._filename) + "_background.png")

        write_data(foreground, "../output/EM/" + str(self._filename) + "_fore.txt")
        read_data("../output/EM/" + str(self._filename) + "_fore.txt", False, save=True, save_name="../output/EM/" + str(self._filename) + "_foreground.png")

        write_data(mask, "../output/EM/" + str(self._filename) + "_mask.txt")
        read_data("../output/EM/" + str(self._filename) + "_mask.txt", False, save=True, save_name="../output/EM/" + str(self._filename) + "_masked.png")

        print('Finished writing data. Please check ' + str(self._filename) + '_background.png, ' + str(self._filename) + '_foreground.png and ' + str(self._filename) + '_masked.png ')

if __name__ == "__main__":
    pictures = ['cow']#['cow', 'fox', 'owl', 'zebra']
    for pic in pictures:
        data, image = read_data('../a2/{}.txt'.format(pic), True)

        img = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))
        clusters, centroids = KMeans(img, k=2)
        # KMeans(n_clusters=2, random_state=0).fit(img)
        # centroids = kmeans.cluster_centers_

        indices_1 = np.random.uniform(0, len(clusters[0]), 3).astype(np.int)
        indices_2 = np.random.uniform(0, len(clusters[1]), 3).astype(np.int)
        cov1 = np.cov([[data[i][2], data[i][3], data[i][4]] for i in indices_1])
        cov2 = np.cov([[data[i][2], data[i][3], data[i][4]] for i in indices_2])

        # cov1 = np.cov([[data[i][2], data[i][3], data[i][4]] for i in range(len(clusters[0]))])
        # cov2 = np.cov([[data[i][2], data[i][3], data[i][4]] for i in range(len(clusters[1]))])

        EM_model = EM(data, image,
                      mean1=centroids[0], mean2=centroids[1],
                      cov1=cov1, cov2=cov2,
                      filename=pic, threshold=1e-5)
        EM_model.execute()
