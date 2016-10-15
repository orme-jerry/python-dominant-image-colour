"""
K-means clustering algorithm for colour detection in images.
"""

import Image
import random
import numpy
import os
import ImageFont
import ImageDraw
from PIL.ExifTags import TAGS
import logging


class Cluster(object):
    def __init__(self):
        self.pixels = []
        self.centroid = None

    def addPoint(self, pixel):
        self.pixels.append(pixel)

    def setNewCentroid(self):
        # creates a 5D centroid by default. x and y elements can be set to 0 when the k-means is run.
        r = [dim[0] for dim in self.pixels]
        g = [dim[1] for dim in self.pixels]
        b = [dim[2] for dim in self.pixels]
        x = [dim[3] for dim in self.pixels]
        y = [dim[4] for dim in self.pixels]

        # deals with empty clusters
        a = list([r, g, b, x, y])
        if len(a[0]) == 0:
            print "Cluster empty!"
            for i in range(0, len(a)):
                a[i] = int(random.sample(range(0, 255), 1)[0])
        else:
            for i in range(0, len(a)):
                a[i] = sum(a[i]) / len(a[i])

        self.centroid = tuple(a)

        self.pixels = []

        return 0

    def centroidColour(self):
        # returns only RGB elements of centroid tuple
        return tuple(self.centroid[:3])


class Photograph(object):
    def __init__(self, directory, filename, suffix, resize=100):
        self.directory = directory
        self.filename = filename
        self.suffix = suffix
        self.image = Image.open(directory + filename + suffix)
        self.resize = (resize, resize)

        # image metadata
        self.ret = {}
        self.metadata = self.image._getexif()
        for tag, value in self.metadata.items():
            decoded = TAGS.get(tag, tag)
            self.ret[decoded] = value

        self.info = self.image.info['exif']

        # thumbnail and pixel array
        self.thumbnailImage = self.image.copy()
        self.thumbnailImage.thumbnail(self.resize)
        self.pixels = numpy.array(self.thumbnailImage.getdata(), dtype=numpy.uint8)

        # watermark colours
        self.most_colourful = None
        self.textColour = None

    def getDateTimeOriginal(self):
        return self.ret['DateTimeOriginal'][0:4]

    # def getOrientation(self):
    #    return self.ret['Orientation']

    def showImage(self):
        self.image.show()

    def getThumbnail(self):
        return self.thumbnailImage

    def showThumbnail(self):
        self.thumbnailImage.show()

    def setMostColourful(self, most_colourful):
        self.most_colourful = most_colourful
        self.setTextColour()

    def setTextColour(self):
        # sets the colour of the overlay text to either white or black for maximum contrast against the background
        # colour
        # if int(numpy.mean(list(self.most_colourful))) < 128:
        #     self.textColour = 255, 255, 255
        # else:
        #     self.textColour = 0, 0, 0

        # sets the colour of the overlay text to either a lighter or darker colour for less contrast against the
        # background colour
        if int(numpy.mean(list(self.most_colourful))) < 128:
            self.textColour = int(round(list(self.most_colourful)[0] * 1.5 + 20)), int(
                round(list(self.most_colourful)[1] * 1.5 + 20)), int(round(list(self.most_colourful)[2] * 1.5 + 20))
        else:
            self.textColour = int(round(list(self.most_colourful)[0] / 1.5 - 20)), int(
                round(list(self.most_colourful)[1] / 1.5 - 20)), int(round(list(self.most_colourful)[2] / 1.5 - 20))

    def watermark(self, ws=0.06):
        # "Watermarks" an image by pasting it into a larger canvas, creating a band along one of the edges, with
        # text.
        # 
        # --------   00000000000    --------000
        # -888888-   00000000000    -888888-000
        # -88-----   00000000000    -88-----000
        # -8888--- + 00000000000 -> -8888---000
        # -88-----   00000000000    -88-----000
        # -88-----   00000000000    -88-----000
        # --------   00000000000    --------000
        
        short_dim = min(self.image.size)
        columns = self.image.size[0]
        rows = self.image.size[1]

        # text height is 0.618 times the width of the band; border is 0.382/2 times the width of the band
        band_size = int(round(short_dim * ws))
        text_size = int(round(0.618 * short_dim * ws))
        border_size = int(round(0.382/2 * short_dim * ws))

        font_path = "/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-C.ttf"
        overlayed_text_1 = "name" + self.getDateTimeOriginal()
        overlayed_text_2 = "website"

        new_canvas = Image.new("RGB", (rows, columns + band_size), self.most_colourful)
        font = ImageFont.truetype(font_path, text_size)

        draw = ImageDraw.Draw(new_canvas)
        text_length = draw.textsize("", font=font)[0]
        draw.text((border_size + int(round(text_length/12)), columns + border_size),
                  overlayed_text_1, self.textColour, font=font)
        draw.text((rows - border_size - text_length, columns + border_size),
                  overlayed_text_2, self.textColour, font=font)
        new_image = new_canvas.rotate(90)
        new_image.paste(self.image, (0, 0))

        return new_image, self.info


class Kmeans(object):
    def __init__(self, k=9, max_iterations=20, min_distance=1.0):
        self.k = k
        self.max_iterations = max_iterations
        self.min_distance = min_distance

    def run(self, photograph, n_dim=5):
        # runs 3D (RGB space) or 5D (RGB-xy space) k-means
        self.pixels = numpy.array(photograph.thumbnailImage.getdata(), dtype=numpy.uint8)
        if n_dim == 3:
            # For 3D k-means, x and y elements are set to 0. NB Centroid is defined in 5D, but optimise for 3D???
            X = Y = list(numpy.repeat(0, photograph.thumbnailImage.size[0] * photograph.thumbnailImage.size[1]))
        elif n_dim == 5:
            # For 5D k-means, x and y elements are assigned each pixel's column and row indices respectively
            X = list(range(photograph.thumbnailImage.size[0]) * photograph.thumbnailImage.size[1])
            x1 = list(range(photograph.thumbnailImage.size[1]))
            Y = list(numpy.repeat(x1, photograph.thumbnailImage.size[0]))
        self.pixels = numpy.array(numpy.column_stack((self.pixels, X, Y)), dtype=numpy.uint8)

        self.clusters = [None for i in range(self.k)]
        self.oldClusters = None

        randomPixels = random.sample(self.pixels, self.k)
        for idx in range(self.k):
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomPixels[idx]

        iterations = 0

        while self.shouldExit(iterations) is False:

            self.oldClusters = [cluster.centroid for cluster in self.clusters]

            print "- Iteration " + str(iterations) + "."

            for pixel in self.pixels:
                self.assignClusters(pixel)

            for cluster in self.clusters:
                cluster.setNewCentroid()

            iterations += 1

        self.passMostColourful(photograph)

        return [cluster.centroid for cluster in self.clusters]

    def assignClusters(self, pixel):
        shortest = float('Inf')
        for cluster in self.clusters:
            distance = self.calcDistance(cluster.centroid, pixel)
            if distance < shortest:
                shortest = distance
                nearest = cluster

        nearest.addPoint(pixel)

    def calcDistance(self, a, b):

        result = numpy.sqrt(sum((a - b) ** 2))
        return result

    def shouldExit(self, iterations):

        if self.oldClusters is None:
            return False

        for idx in range(self.k):
            dist = self.calcDistance(
                numpy.array(self.clusters[idx].centroid),
                numpy.array(self.oldClusters[idx])
            )
            if dist < self.min_distance:
                return True

        if iterations <= self.max_iterations:
            return False

        return True

    def showCentroidColours(self):
        # creates a 700*700 px canvas to display all k centroid colours in grid format
        elementSize = int(round((700 / numpy.ceil(numpy.sqrt(self.k)))))
        gridSize = int(elementSize * numpy.ceil(numpy.sqrt(self.k)))
        canvas = Image.new("RGB", (gridSize, gridSize), (255, 255, 255))
        x = y = 0
        for cluster in self.clusters:
            image = Image.new("RGB", (elementSize, elementSize), cluster.centroidColour())
            canvas.paste(image, (x * elementSize, y * elementSize))
            if x < 2:
                x += 1
            else:
                x = 0
                y += 1

        return canvas

    # def showClustering(self):
    # 
    #     localPixels = [None] * len(self.image.getdata())
    # 
    #     for idx, pixel in enumerate(self.pixels):
    #         shortest = float('Inf')
    #         for cluster in self.clusters:
    #             distance = self.calcDistance(cluster.centroid, pixel)
    #             if distance < shortest:
    #                 shortest = distance
    #                 nearest = cluster
    # 
    #         localPixels[idx] = nearest.centroid
    # 
    #     w, h = self.image.size
    #     localPixels = numpy.asarray(localPixels) \
    #         .astype('uint8') \
    #         .reshape((h, w, 3))
    # 
    #     colourMap = Image.fromarray(localPixels)
    #     colourMap.show()

    def passMostColourful(self, photograph):
        # finds the "most colourful" centroid and passes its RGB values back to Photograph(). The "most colourful"
        # centroid is interpreted as the furthest from the grey diagonal in RGB space. It is calculated as the cluster
        # with the largest range of RGB components (i.e. largest (max(R, G, B) - min(R, G, B)).
        most_colourful = (0, 0, 0)
        for cluster in self.clusters:
            if max(list(cluster.centroidColour())) - min(list(cluster.centroidColour())) > max(
                    list(most_colourful)) - min(
                list(most_colourful)):
                most_colourful = cluster.centroidColour()
        photograph.setMostColourful(most_colourful)


def main():
    directory = ""
    suffix = ".jpg"
    name_exts = os.listdir(directory)
    filenames = [os.path.splitext(n)[0] for n in name_exts]
    print filenames

    if os.path.isdir(directory + "watermarked/"):
        pass
    else:
        os.makedirs(directory + "watermarked/")

    for filename in filenames:
        print filename
        if os.path.isfile(directory + "watermarked/watermarked_" + filename + suffix):
            print "- File already exists. No action taken."
        else:
            #for i in range(1, 20):
            p = Photograph(directory, filename, suffix)
            k = Kmeans()
            k.run(p)
            #k.showCentroidColours().save(directory + "watermarked/" + str(i) + "_5D_centroids_" + filename + suffix, "JPEG",
            #                          quality=100)
            new_image = p.watermark()
            new_image[0].save(directory + "watermarked/" + "watermarked_" + filename + suffix, "JPEG",
                                  quality=100, exif=new_image[1])


if __name__ == "__main__":
    main()
