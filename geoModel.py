import os
import vtk
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
import matplotlib.pyplot as plt

import rasterio
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import Polygon, Point

from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

import joblib


class geoModel():
    def __init__(self):
        self.fileDict = {}
        self.locDict = {}
        self.litoDict = {}
        self.litoFileDict = {}

    def defineFileDict(self, fileDict):
        self.fileDict = fileDict

    def defineLitoDict(self, litoDict):
        self.litoDict = litoDict

    def defineLocDict(self, locDict):
        self.locDict = locDict

    def defineLitoFileDict(self, litoFileDict):
        self.litoFileDict = litoFileDict

    def generatePointCloud(self, resolution=0.3):
        # import well location
        wellLoc = pd.read_csv(self.fileDict['locationFile'])
        wellLoc = wellLoc.set_index(self.locDict['id'])

        # import borehole litology
        wellLito = pd.read_csv(self.fileDict['litoFile'])

        # create empty columns
        wellLito['elevTop'] = pd.Series(-10000.0, index=wellLito.index, dtype='float64')
        wellLito['elevBot'] = pd.Series(-10000.0, index=wellLito.index, dtype='float64')

        for index, row in wellLito.iterrows():
            # print(row)
            try:
                surfElev = wellLoc.loc[row[self.litoFileDict['id']], self.locDict['elevation']]
                wellLito.loc[index, 'elevTop'] = surfElev - row[self.litoFileDict['top']]
                wellLito.loc[index, 'elevBot'] = surfElev - row[self.litoFileDict['bottom']]
            except KeyError:
                wellLito = wellLito.drop(index)

        # check well lito and export as csv
        litoName = self.fileDict['locationFile'].split('.')[0]
        # litoElevFile = os.path.join(self.fileDict['outputDir'],litoName+'_Elev.csv')
        # wellLito.to_csv(litoElevFile)
        # self.fileDict['litoElevFile'] = litoElevFile

        # store dataframes for later use
        self.wellLitoDf = wellLito
        self.wellLocDf = wellLoc

        litoPoints = []

        for index, values in self.wellLitoDf.iterrows():
            id = self.locDict['id']
            easting = self.locDict['easting']
            northing = self.locDict['northing']
            elevation = self.locDict['elevation']
            wellX, wellY, wellZ = self.wellLocDf.loc[values[self.litoFileDict['id']]][[easting, northing, elevation]]
            wellXY = [wellX, wellY]
            litoPoints.append(wellXY + [values.elevTop, values[self.litoFileDict['litoCode']]])
            litoPoints.append(wellXY + [values.elevBot, values[self.litoFileDict['litoCode']]])

            litoLength = values.elevTop - values.elevBot

            depthResolution = resolution
            if litoLength < depthResolution:
                midPoint = wellXY + [values.elevTop - litoLength / 2, values[self.litoFileDict['litoCode']]]
            else:
                npoints = int(litoLength / depthResolution)
                for point in range(1, npoints):
                    disPoint = wellXY + [values.elevTop - litoLength * point / npoints,
                                         values[self.litoFileDict['litoCode']]]
                    litoPoints.append(disPoint)
        self.litoPointCloud = np.array(litoPoints)
        pointCloudFile = os.path.join(self.fileDict['outputDir'], 'litoPointCloud')
        np.save(pointCloudFile, self.litoPointCloud)

    def generateLitoRepresentation(self):
        # generation of list arrays for the vtk
        offsetList = []
        linSec = []
        linVerts = []

        i = 0
        for index, values in self.wellLitoDf.iterrows():
            x, y = self.wellLocDf.loc[values[self.locDict['id']]][[self.locDict['easting'], self.locDict['northing']]]
            cellVerts = [[x, y, values.elevTop], [x, y,
                                                  values.elevBot]]
            # print(cellVerts)
            offsetList.append(i * 3)
            linSec = linSec + [2, 2 * i, 2 * i + 1]
            linVerts = linVerts + cellVerts
            i += 1

        offsetArray = np.array(offsetList)
        linArray = np.array(linSec)
        cellType = np.ones([i]) * 3
        vertArray = np.array(linVerts)
        # create the unstructured grid and assign lito code
        grid = pv.UnstructuredGrid(linArray, cellType, vertArray)
        grid.cell_data["values"] = self.wellLitoDf[self.litoFileDict['litoCode']].values
        litoVtuFile = os.path.join(self.fileDict['outputDir'], 'conceptualizedLito.vtu')
        grid.save(litoVtuFile, binary=False)

    def buildNeuralClassifier(self, max_iter=3000):
        # transform to local coordinates
        self.litoMean = self.litoPointCloud[:, :3].mean(axis=0)
        self.litoTrans = self.litoPointCloud[:, :3] - self.litoMean

        # setting up scaler
        self.scaler = preprocessing.StandardScaler().fit(self.litoTrans)

        # define x and y
        X = self.scaler.transform(self.litoTrans)
        y = self.litoPointCloud[:, 3]

        # split in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # run classifier
        self.clf = MLPClassifier(activation='tanh',
                                 solver='adam',
                                 hidden_layer_sizes=(15, 15, 15, 15, 15),
                                 alpha=0.001,
                                 batch_size=256,
                                 max_iter=max_iter,
                                 learning_rate_init=0.001,
                                 random_state=42,
                                 verbose=True)
        print("Rozpoczynam trenowanie L-BFGS...")
        self.clf.fit(X_train, y_train)

        predicted = self.clf.predict(X_test)

        self.report(y_test, predicted)

    def buildNeuralClassifierSMOT(self, max_iter=3000):
        # 1. Przygotowanie danych (X, y)
        self.litoMean = self.litoPointCloud[:, :3].mean(axis=0)
        self.litoTrans = self.litoPointCloud[:, :3] - self.litoMean


        self.scaler = preprocessing.StandardScaler().fit(self.litoTrans)

        X = self.scaler.transform(self.litoTrans)
        y = self.litoPointCloud[:, 3]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # 2. SMOTE - Balansowanie tylko zbioru treningowego
        print(f"Oryginalny rozkład klas: {np.unique(y_train, return_counts=True)}")
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print(f"Zbalansowany rozkład klas: {np.unique(y_train_res, return_counts=True)}")

        # 3. Model LBFGS
        self.clf = MLPClassifier(activation='tanh',
                                 solver='adam',
                                 hidden_layer_sizes=(60, 40, 20),
                                 alpha=0.001,
                                 batch_size=256,
                                 max_iter=max_iter,
                                 learning_rate_init=0.001,
                                 random_state=42,
                                 verbose=True
                                 )

        print("Rozpoczynam trenowanie L-BFGS na zbalansowanych danych...")
        self.clf.fit(X_train_res, y_train_res)

        # 4. Predykcja i wyniki
        predicted = self.clf.predict(X_test)

        self.report(y_test, predicted)

    def buildNeuralClassifierSMOTNewFeauture(self, max_iter=3000):
        # 1. Przygotowanie danych wejściowych
        self.litoMean = self.litoPointCloud[:, :3].mean(axis=0)
        self.litoTrans = self.litoPointCloud[:, :3] - self.litoMean
        X_raw = self.litoTrans  # Współrzędne x, y, z (wyśrodkowane)

        # --- FEATURE ENGINEERING ---
        # Dodajemy odległość horyzontalną (promień) - pomaga sieciom zrozumieć strukturę walcową otworu
        dist_xy = np.sqrt(X_raw[:, 0] ** 2 + X_raw[:, 1] ** 2)
        # Kwadrat głębokości - pomaga wyłapać nieliniowe granice warstw
        z_sq = X_raw[:, 2] ** 2

        # Łączymy cechy w jedną macierz
        X_enhanced = np.column_stack((X_raw, dist_xy, z_sq))
        y = self.litoPointCloud[:, 3]

        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y, test_size=0.33, random_state=42)

        # 2. Skalowanie (kluczowe po dodaniu z_sq, który ma inne rzędy wielkości)
        self.scaler = preprocessing.StandardScaler().fit(X_enhanced)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 3. SMOTE - Balansowanie klas
        print(f"Oryginalny rozkład klas: {np.unique(y_train, return_counts=True)}")
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
        print(f"Zbalansowany rozkład klas: {np.unique(y_train_res, return_counts=True)}")

        # 4. Model - Przechodzimy na 'adam', ale z szerszą architekturą
        # Szersze warstwy (60, 40) lepiej radzą sobie z separacją klas niż wiele wąskich (15, 15...)
        self.clf = MLPClassifier(activation='tanh',
                                 solver='adam',
                                 hidden_layer_sizes=(60, 40, 20),
                                 alpha=0.001,
                                 batch_size=256,
                                 max_iter=max_iter,
                                 learning_rate_init=0.001,
                                 random_state=42,
                                 verbose=True)

        print(f"Rozpoczynam trenowanie na zbalansowanym zbiorze: {X_train_res.shape[0]} punktów...")
        self.clf.fit(X_train_res, y_train_res)

        # 5. Wyniki i raport
        predicted = self.clf.predict(X_test_scaled)

        self.report(y_test, predicted)

    def report(self, y_test, predicted):
        # Wyświetlenie raportu
        print("\nRaport:")
        print(classification_report(y_test, predicted))

        # 3. MACIERZ POMYŁEK (Twoja dotychczasowa, ulepszona)
        results = confusion_matrix(y_test, predicted)
        litoDf = pd.DataFrame(self.litoDict, index=[1])

        plt.subplot(1, 2, 2)
        sns.heatmap(results,
                    annot=True,
                    fmt="d",
                    cmap="coolwarm",
                    yticklabels=litoDf.columns.values.tolist(),
                    xticklabels=litoDf.columns.values.tolist())
        plt.title(f'Macierz pomyłek\nAccuracy: {accuracy_score(y_test, predicted):.2f}')

        plt.tight_layout()
        plt.show()

    def generatePredictedGrid(self, cellHeight=0.3, cellWidth=20, dem_path=None):
        """
        Generuje siatkę litologiczną wykorzystując model ML z 5 cechami (X, Y, Z, dist, z^2).
        Przycina wynik do powierzchni terenu na podstawie GeoTIFF.
        Wyniki zapisuje w atrybutach obiektu (self.litoMatrix, self.litoMatrixMod,
        self.vertexCols, self.vertexRows, self.vertexLays, self.cellCols, self.cellRows, self.cellLays).
        """

        # Jeśli nie podano ścieżki w argumencie, weź ze słownika plików
        if dem_path is None:
            dem_path = self.fileDict.get('demFile', 'sciezka/do/dem.tif')

        # 1. Definicja zakresów
        xMin = self.wellLocDf[self.locDict['easting']].min() // 100 * 100
        xMax = self.wellLocDf[self.locDict['easting']].max() // 100 * 100
        yMin = self.wellLocDf[self.locDict['northing']].min() // 100 * 100
        yMax = self.wellLocDf[self.locDict['northing']].max() // 100 * 100

        # Bufor bezpieczeństwa dla Z
        zMax = round(self.wellLitoDf['elevTop'].max(), 1) + 5.0
        zMin = round(self.wellLitoDf['elevBot'].min(), 1) - 5.0

        cellH = cellWidth
        cellV = cellHeight

        # Siatki współrzędnych
        self.vertexCols = np.arange(xMin, xMax + cellH, cellH)
        self.vertexRows = np.arange(yMax, yMin - cellH, -cellH)
        self.vertexLays = np.arange(zMax, zMin - cellV, -cellV)

        self.cellCols = (self.vertexCols[1:] + self.vertexCols[:-1]) / 2
        self.cellRows = (self.vertexRows[1:] + self.vertexRows[:-1]) / 2
        self.cellLays = (self.vertexLays[1:] + self.vertexLays[:-1]) / 2

        nCols = len(self.cellCols)
        nRows = len(self.cellRows)
        nLays = len(self.cellLays)

        print('/---------- Grid Parameters ----------/')
        print(f"Grid dims: {nCols} x {nRows} x {nLays} (Total cells: {nCols * nRows * nLays})")

        AIR_VALUE = -1.0
        self.litoMatrix = np.full((nLays, nRows, nCols), AIR_VALUE, dtype=float)

        # 2. Przygotowanie siatki XY (Meshgrid)
        XX, YY = np.meshgrid(self.cellCols, self.cellRows)

        flat_X = XX.flatten()
        flat_Y = YY.flatten()

        # 3. Pobranie DEM z GeoTIFF
        print("Próbkowanie DEM z GeoTIFF...")

        try:
            with rasterio.open(dem_path) as src:
                coords = list(zip(flat_X, flat_Y))
                flat_dem_z = np.fromiter((val[0] for val in src.sample(coords)), dtype=float)

                if src.nodata is not None:
                    flat_dem_z[flat_dem_z == src.nodata] = -9999

        except Exception as e:
            print(f"BŁĄD ODCZYTU DEM: {e}")
            print("Przyjmuję płaski teren na Z=maximum.")
            flat_dem_z = np.full_like(flat_X, zMax)

        print("Rozpoczynam predykcję warstwa po warstwie...")

        mean_x, mean_y, mean_z = self.litoMean[:3]

        # 4. Główna pętla po warstwach
        for i_lay, z_level in enumerate(self.cellLays):

            flat_Z = np.full_like(flat_X, z_level)
            mask_underground = flat_Z <= flat_dem_z

            if not np.any(mask_underground):
                continue

            X_active = flat_X[mask_underground]
            Y_active = flat_Y[mask_underground]
            Z_active = flat_Z[mask_underground]

            # Feature engineering (identyczne jak przy treningu)
            X_trans = X_active - mean_x
            Y_trans = Y_active - mean_y
            Z_trans = Z_active - mean_z

            dist_xy = np.sqrt(X_trans ** 2 + Y_trans ** 2)
            z_sq = Z_trans ** 2

            points_enhanced = np.column_stack((X_trans, Y_trans, Z_trans, dist_xy, z_sq))
            points_scaled = self.scaler.transform(points_enhanced)

            preds = self.clf.predict(points_scaled)

            layer_result = np.full(nCols * nRows, AIR_VALUE)
            layer_result[mask_underground] = preds
            self.litoMatrix[i_lay, :, :] = layer_result.reshape(nRows, nCols)

            if i_lay % 5 == 0:
                pct = (i_lay / nLays) * 100
                print(f"Progress: {pct:.1f}% (Warstwa Z={z_level:.1f})")

        print('/---------- Grid generation complete ----------/')

        plt.imshow(self.litoMatrix[10])

        self.litoMatrixMod = self.litoMatrix[::-1, ::-1, ::-1]

    def saveGridToVtk(self, filename='predictedGeology.vtk'):
        """
        Zapisuje wygenerowaną siatkę litologiczną do pliku VTK (do wizualizacji w ParaView).
        Wymaga wcześniejszego wywołania generatePredictedGrid().
        """
        grid = pv.RectilinearGrid(self.vertexCols, self.vertexRows, self.vertexLays)

        litoFlat = list(self.litoMatrixMod.flatten(order="K"))[::-1]
        grid.cell_data["geoCode"] = np.array(litoFlat)
        grid = grid.threshold(value=0.5, scalars="geoCode", preference="cell")
        gridVtuFile = os.path.join(self.fileDict['outputDir'], filename)
        grid.save(gridVtuFile)
        print(f"Grid VTK zapisany do {gridVtuFile}")

    def saveGridToNpz(self, filename='predictedGrid.npz'):
        """
        Zapisuje wygenerowaną siatkę litologiczną do pliku NPZ (do użycia w Streamlit).
        Wymaga wcześniejszego wywołania generatePredictedGrid().

        Plik NPZ zawiera:
          - cellCols, cellRows, cellLays: współrzędne centrów komórek
          - litoMatrix: macierz 3D (nLays, nRows, nCols) z kodami litologii (-1 = powietrze)
        """
        npzFile = os.path.join(self.fileDict['outputDir'], filename)
        np.savez(npzFile,
                 cellCols=self.cellCols,
                 cellRows=self.cellRows,
                 cellLays=self.cellLays,
                 litoMatrix=self.litoMatrix)
        print(f"Grid NPZ zapisany do {npzFile}")

    def saveModel(self, filename='neuralModel.joblib'):
        # Musisz zapisać zarówno model, jak i scaler!
        # Bez scalera nie będziesz mógł poprawnie przygotować nowych danych.
        model_data = {
            'clf': self.clf,
            'scaler': self.scaler,
            'litoMean': self.litoMean
        }
        modelFile = os.path.join(self.fileDict['outputDir'], filename)
        joblib.dump(model_data, modelFile)
        print(f"Model zapisany do {modelFile}")

    def loadModel(self, filename='neuralModel.joblib'):
        modelFile = os.path.join(self.fileDict['outputDir'], filename)
        model_data = joblib.load(modelFile)
        self.clf = model_data['clf']
        self.scaler = model_data['scaler']
        self.litoMean = model_data['litoMean']
        print("Model wczytany pomyślnie")

    def predictLithology(self, x, y, z):
        """
        Predict lithology for a single point (x, y, z).
        Applies the same feature engineering as buildNeuralClassifierSMOTNewFeauture:
        centering, dist_xy, z².
        """
        mean_x, mean_y, mean_z = self.litoMean[:3]
        x_trans = x - mean_x
        y_trans = y - mean_y
        z_trans = z - mean_z
        dist_xy = np.sqrt(x_trans ** 2 + y_trans ** 2)
        z_sq = z_trans ** 2

        features = np.array([[x_trans, y_trans, z_trans, dist_xy, z_sq]])
        features_scaled = self.scaler.transform(features)
        return self.clf.predict(features_scaled)



