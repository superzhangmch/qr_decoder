"""Galois Field GF(2^8) and Reed-Solomon error correction with erasure support."""


class GF:
    """Galois Field GF(2^8) for Reed-Solomon."""
    def __init__(self):
        self.exp, self.log = [0]*512, [0]*256
        x = 1
        for i in range(255):
            self.exp[i], self.log[x] = x, i
            x = (x << 1) ^ 0x11D if x & 0x80 else x << 1
        for i in range(255, 512):
            self.exp[i] = self.exp[i - 255]

    def mul(self, a, b):
        return 0 if a == 0 or b == 0 else self.exp[self.log[a] + self.log[b]]

    def div(self, a, b):
        return 0 if a == 0 else self.exp[(self.log[a] - self.log[b]) % 255]

    def inv(self, a):
        return self.exp[255 - self.log[a]]


class ReedSolomon:
    """Reed-Solomon error correction with erasure support."""
    def __init__(self, nsym):
        self.nsym, self.gf = nsym, GF()

    def _poly_mul(self, p1, p2):
        r = [0] * (len(p1) + len(p2) - 1)
        for i, c1 in enumerate(p1):
            for j, c2 in enumerate(p2):
                r[i + j] ^= self.gf.mul(c1, c2)
        return r

    def _poly_eval(self, p, x):
        r = 0
        for c in p:
            r = self.gf.mul(r, x) ^ c
        return r

    def _syndromes(self, msg):
        return [self._poly_eval(msg, self.gf.exp[i]) for i in range(self.nsym)]

    def _berlekamp_massey(self, syndromes):
        """Berlekamp-Massey algorithm to find error locator polynomial."""
        n = len(syndromes)
        C = [1]
        B = [1]
        L = 0
        m = 1
        b = 1

        for n_i in range(n):
            d = syndromes[n_i]
            for i in range(1, L + 1):
                if i < len(C):
                    d ^= self.gf.mul(C[i], syndromes[n_i - i])

            if d == 0:
                m += 1
            elif 2 * L <= n_i:
                T = C.copy()
                C = C + [0] * (m - (len(C) - len(B)))
                for i in range(len(B)):
                    C[i + m] ^= self.gf.mul(self.gf.div(d, b), B[i])
                L = n_i + 1 - L
                B = T
                b = d
                m = 1
            else:
                C = C + [0] * max(0, len(B) + m - len(C))
                for i in range(len(B)):
                    C[i + m] ^= self.gf.mul(self.gf.div(d, b), B[i])
                m += 1

        return C

    def _forney_syndromes(self, syndromes, erasure_pos, n):
        """Compute Forney syndromes for erasure+error correction."""
        fsynd = list(syndromes)
        for pos in erasure_pos:
            x = self.gf.exp[n - 1 - pos]
            for i in range(len(fsynd) - 1):
                fsynd[i] = self.gf.mul(fsynd[i], x) ^ fsynd[i + 1]
        return fsynd[:-len(erasure_pos)] if erasure_pos else fsynd

    def _find_errors(self, err_loc, n):
        return [n - 1 - i for i in range(n) if self._poly_eval(err_loc, self.gf.exp[i]) == 0]

    def _correct(self, msg, synd, pos):
        if not pos: return msg
        # Build standard error locator Λ(x) = Π(1 + X_j*x), descending order
        err_loc = [1]
        for p in pos:
            err_loc = self._poly_mul(err_loc, [self.gf.exp[len(msg) - 1 - p], 1])
        omega = self._poly_mul(synd[::-1], err_loc)[-(self.nsym):]
        # Formal derivative in GF(2^8): d/dx(a*x^k) = a*x^{k-1} if k odd, 0 if k even
        n = len(err_loc) - 1
        deriv = [err_loc[i] if (n - i) % 2 == 1 else 0 for i in range(n)] or [0]
        msg = list(msg)
        for p in pos:
            Xi = self.gf.exp[len(msg) - 1 - p]
            d = self._poly_eval(deriv, self.gf.inv(Xi))
            if d != 0:
                msg[p] ^= self.gf.mul(Xi, self.gf.div(self._poly_eval(omega, self.gf.inv(Xi)), d))
        return msg

    def decode(self, msg, erasure_pos=None):
        """Decode with optional erasure positions.

        With erasures: can correct 2*errors + erasures <= nsym
        Without erasures: can correct errors <= nsym/2
        """
        erasure_pos = erasure_pos or []
        synd = self._syndromes(msg)
        if max(synd) == 0: return msg[:-self.nsym]

        if erasure_pos:
            # Erasure + error correction using Forney syndromes
            if len(erasure_pos) > self.nsym:
                raise ValueError("Too many erasures")
            fsynd = self._forney_syndromes(synd, erasure_pos, len(msg))
            err_loc = self._berlekamp_massey(fsynd) if fsynd else [1]
            num_errors = len(err_loc) - 1
            if 2 * num_errors + len(erasure_pos) > self.nsym:
                raise ValueError("Too many errors+erasures")
            err_pos = self._find_errors(err_loc, len(msg)) if num_errors > 0 else []
            all_pos = list(set(erasure_pos + err_pos))
        else:
            err_loc = self._berlekamp_massey(synd)
            if len(err_loc) - 1 > self.nsym // 2: raise ValueError("Too many errors")
            all_pos = self._find_errors(err_loc, len(msg))
            if len(all_pos) != len(err_loc) - 1: raise ValueError("Cannot locate errors")

        corrected = self._correct(msg, synd, all_pos)
        if max(self._syndromes(corrected)) != 0: raise ValueError("Correction failed")
        return corrected[:-self.nsym]
