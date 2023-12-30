# -*- coding: utf-8 -*-

import numpy as np

Pi = np.pi

class Schordinger(object):

    def __init__(self) -> None:

        self.L = 5.0
        self.a1 = 0.0
        self.b1 = self.a1 + self.L
        self.coef_a = 0.5

        self.name = "oned-schrodinger"

        self.vanal_ur = np.vectorize(self.anal_ur)
        self.vanal_ui = np.vectorize(self.anal_ui)
        self.vanal_gr = np.vectorize(self.anal_gr)
        self.vanal_gi = np.vectorize(self.anal_gi)
        self.vanal_fr = np.vectorize(self.anal_fr)
        self.vanal_fi = np.vectorize(self.anal_fi)

    def anal_ur(self, x, t):

        omega = 2 * Pi / self.L
        ut = np.cos(- self.coef_a * np.power(omega, 2.0) * t)
        ux = 2 * np.cos(omega * x) + np.sin(omega * x)

        u = ut * ux

        return u

    def anal_ui(self, x, t):

        omega = 2 * Pi / self.L
        vt = np.sin(- self.coef_a * np.power(omega, 2.0) * t)
        vx = 2 * np.cos(omega * x) + np.sin(omega * x)

        v = vt * vx

        return v

    def anal_gr(self, x, t=0):

        phi = self.anal_ur(x, 0)

        return phi

    def anal_gi(self, x, t=0):

        psi = self.anal_ui(x, 0)

        return psi

    def anal_fr(self, x, t):

        return 0

    def anal_fi(self, x, t):

        return 0