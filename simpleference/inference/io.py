# try to import z5py
try:
    import z5py
    WITH_Z5PY = True
except ImportError:
    WITH_Z5PY = False

# try to import h5py
try:
    import h5py
    WITH_H5PY = True
except ImportError:
    WITH_H5PY = False

# try to import dvid
try:
    from libdvid import DVIDNodeService
    from libdvid import ConnectionMethod
    WITH_DVID = True
except ImportError:
    WITH_DVID = False


class IoN5(object):
    def __init__(self, path, keys, save_only_nn_affs=False):
        assert WITH_Z5PY, "Need z5py"
        assert len(keys) in (1, 2)
        self.path = path
        if save_only_nn_affs:
            assert len(keys) == 2
        self.save_only_nn_affs = save_only_nn_affs
        self.keys = keys
        self.ff = z5py.File(self.path, use_zarr_format=False)
        assert all(kk in self.ff for kk in self.keys)
        self.datasets = [self.ff[kk] for kk in self.keys]
        # we just assume that everything has the same shape...
        self.shape = self.datasets[0].shape

    def read(self, bounding_box):
        assert len(self.datasets) == 1
        return self.datasets[0][bounding_box]

    def write(self, out, out_bb):
        if self.save_only_nn_affs:
            self._write_nn_affs(out, out_bb)
        else:
            self._write_all(out, out_bb)

    def _write_nn_affs(self, out, out_bb):
        assert len(self.datasets) == 1
        bb = (slice(None),) + out_bb
        self.datasets[0][bb] = out

    def _write_all(self, out, out_bb):
        self.datasets[0][out_bb] = (out[1] + out[2]) / 2.
        self.datasets[1][out_bb] = out[0]

    @property
    def shape(self):
        return self.shape

    def close(self):
        pass


class IoHDF5(object):
    def __init__(self, path, key):
        assert WITH_H5PY, "Need h5py"
        self.path = path
        self.ff = h5py.File(self.path)
        self.ds = self.ff[key]
        self.shape = self.ds.shape

    def read(self, bb):
        return self.ds[bb]

    def write(self, out, out_bb):
        bb = (slice(None),) + out_bb
        self.ds[bb] = out

    @property
    def shape(self):
        return self.shape

    def close(self):
        self.ff.close()


class IoDVID(object):
    def __init__(self, server_address, uuid, key):
        assert WITH_DVID, "Need dvid"
        self.ds = DVIDNodeService(server_address, uuid)
        self.key = key

        # get the shape the dvid way...
        endpoint = "/" + self.key + "/info"
        attributes = self.ds.custom_request(endpoint, "", ConnectionMethod.GET)
        # TODO do we need to increase by 1 here
        self.shape = tuple(mp + 1 for mp in attributes["MaxPoint"])

    def read(self, bb):
        offset = tuple(b.start for b in bb)
        shape = tuple(b.stop - b.start for b in bb)
        return self.ds.get_gray3D(self.key, shape, offset)

    def write(self, out, out_bb):
        raise NotImplementedError("Writing to DVID is not yet implemented!")

    @property
    def shape(self):
        return self.shape

    def close(self):
        pass
