import numpy as np
import math

class Transform():
    def __init__(self):
        self.a = 6378137.0  # semi-major axis, meters
        self.f = 1 / 298.257223563  # flattening
        self.e2 = 2 * self.f - self.f ** 2  # eccentricity squared

        self.ref_lla = [10.87050, 106.802, 0]

    def lla_to_ecef(self, lat, lon, alt):
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)

        N = self.a / np.sqrt(1 - self.e2 * np.sin(lat) ** 2)

        X = (N + alt) * np.cos(lat) * np.cos(lon)
        Y = (N + alt) * np.cos(lat) * np.sin(lon)
        Z = (N * (1 - self.e2) + alt) * np.sin(lat)

        return np.array([X, Y, Z])

    def ecef_to_ned(self, ecef):
        ref_ecef = self.lla_to_ecef(*self.ref_lla)
        dX = ecef - ref_ecef

        lat, lon = np.deg2rad(self.ref_lla[0]), np.deg2rad(self.ref_lla[1])
        R = np.array([[-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                    [-np.sin(lon), np.cos(lon), 0],
                    [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]])

        ned = R @ dX
        return ned

    def ned_to_ecef(self, ned):
        ref_ecef = self.lla_to_ecef(*self.ref_lla)
        lat, lon = np.deg2rad(self.ref_lla[0]), np.deg2rad(self.ref_lla[1])

        R = np.array([[-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                    [-np.sin(lon), np.cos(lon), 0],
                    [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]])

        dX = np.linalg.inv(R) @ ned
        ecef = ref_ecef + dX

        return ecef

    def ecef_to_lla(self, ecef):
        X, Y, Z = ecef
        lon = np.arctan2(Y, X)

        p = np.sqrt(X ** 2 + Y ** 2)
        lat = np.arctan2(Z, p * (1 - self.e2))
        alt = 0

        for _ in range(5):
            N = self.a / np.sqrt(1 - self.e2 * np.sin(lat) ** 2)
            alt = p / np.cos(lat) - N
            lat = np.arctan2(Z, p * (1 - self.e2 * N / (N + alt)))

        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)

        return np.array([lat, lon, alt])

    def lla_to_ned(self, lla):
        ecef = self.lla_to_ecef(*lla)
        ned = self.ecef_to_ned(ecef)
        return ned

    def ned_to_lla(self, ned):
        ecef = self.ned_to_ecef(ned)
        lla = self.ecef_to_lla(ecef)
        return lla