#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif


void nb_comparison(PyArrayObject *npsingles, PyArrayObject *npcompare, PyArrayObject *priors,
                     PyArrayObject *likelihood, PyArrayObject *results) {
	int t, cl, kc;
	int ncl = PyArray_DIM(npsingles, 0), nkc = PyArray_DIM(npsingles, 1), nt = PyArray_DIM(npcompare, 1);
	double denom, product, compkc;

	#pragma omp parallel shared(npsingles, npcompare, priors, nkc, nt, results) private(t, cl, kc, denom, product, compkc)
	{
		#pragma omp for schedule(dynamic, 100)
		// Iterate over all time points
		for (t = 0; t < nt; t++) {
		//for (t = 0; t < 100; t++) {
			// Divide by the denominator
			denom = 0.0;
			for (cl = 0; cl < ncl; cl++) {
				product = 1.0;
				for (kc = 0; kc < nkc; kc++) {
				    // Prob spiking*probability of spiking during condition +
					compkc = *(double *) PyArray_GETPTR2(npcompare, kc, t);
					product *= *(double *) PyArray_GETPTR3(npsingles, cl, kc, 0)*compkc +
						*(double *) PyArray_GETPTR3(npsingles, cl, kc, 1)*(1.0 - compkc);
				}

                *(double *)PyArray_GETPTR2(likelihood, cl, t) = product;
				*(double *)PyArray_GETPTR2(results, cl, t) = *(double *) PyArray_GETPTR2(priors, cl, t)*product;
                denom += *(double *)PyArray_GETPTR2(results, cl, t);
			}

			for (cl = 0; cl < ncl; cl++) {
				*(double *)PyArray_GETPTR2(results, cl, t) /= denom;
			}
		}
	}
}

static PyObject* naivebayes(PyObject *dummy, PyObject *args) {
	PyObject *arg1=NULL, *arg3=NULL, *arg4=NULL, *arg5=NULL, *arg6=NULL;
	PyArrayObject *nps=NULL, *npcl=NULL, *npcomp=NULL, *npresults=NULL, *nplike=NULL;

	// Parse the input from NumPy
	import_array();
	if (!PyArg_ParseTuple(args, "OOOO!O!", &arg1, &arg3, &arg4,
		&PyArray_Type, &arg5, &PyArray_Type, &arg6)) return NULL;

	nps = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	npcl = (PyArrayObject *)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	npcomp = (PyArrayObject *)PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	npresults = (PyArrayObject *)PyArray_FROM_OTF(arg5, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
	nplike = (PyArrayObject *)PyArray_FROM_OTF(arg6, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

	// Fail if necessary
	if (nps == NULL || npcomp == NULL ||
		npcl == NULL || npresults == NULL || nplike == NULL) goto fail;

	if (PyArray_NDIM(npcomp) != 2) {PySys_WriteStdout("ERROR: Comparison data has wrong number of dimensions\n"); goto fail;};
	if (PyArray_NDIM(nps) != 3) {PySys_WriteStdout("ERROR: Singles have wrong number of dimensions\n"); goto fail;};
	if (PyArray_NDIM(npcl) != 2) {PySys_WriteStdout("ERROR: Priors have wrong number of dimensions\n"); goto fail;};
	if (PyArray_ITEMSIZE(npcl) != 8) {PySys_WriteStdout("ERROR: Priors is not of type np.float64\n"); goto fail;};

	// Run
	nb_comparison(nps, npcomp, npcl, nplike, npresults);

	Py_DECREF(nps);
	Py_DECREF(npcl);
	Py_DECREF(npcomp);
	Py_DECREF(npresults);
	Py_DECREF(nplike);
    Py_INCREF(Py_None);
    return Py_None;

    // Account for failure by setting error and returning null
    fail:
	    Py_XDECREF(nps);
		Py_XDECREF(npcl);
		Py_XDECREF(npcomp);
		PyArray_XDECREF_ERR(npresults);
		PyArray_XDECREF_ERR(nplike);
		return NULL;
}

/*void nb_comparison(PyArrayObject *npsingles, PyArrayObject *npcompare, PyArrayObject *priors,
                     PyArrayObject *likelihood, PyArrayObject *results) {
	int t, cl, kc;
	int ncl = PyArray_DIM(npsingles, 0), nkc = PyArray_DIM(npsingles, 1), nt = PyArray_DIM(npcompare, 1);
	double denom, product, compkc;

	#pragma omp parallel shared(npsingles, npcompare, priors, ncl, nkc, nt, results) private(t, cl, kc, denom, product, compkc)
	{
		#pragma omp for schedule(dynamic, 100)
		// Iterate over all time points
		for (t = 0; t < nt; t++) {
		//for (t = 0; t < 100; t++) {
			// Divide by the denominator
			denom = 0.0;
			for (cl = 0; cl < ncl; cl++) {
				product = 0.0;
				for (kc = 0; kc < nkc; kc++) {
				    // Prob spiking*probability of spiking during condition +
					compkc = *(double *) PyArray_GETPTR2(npcompare, kc, t);
					product *= *(double *) PyArray_GETPTR3(npsingles, cl, kc, 0)*compkc +
						*(double *) PyArray_GETPTR3(npsingles, cl, kc, 1)*(1.0 - compkc);
				}

                *(double *)PyArray_GETPTR2(likelihood, cl, t) = product;
				*(double *)PyArray_GETPTR2(results, cl, t) = *(double *) PyArray_GETPTR2(priors, cl, t)*product;
                denom += *(double *)PyArray_GETPTR2(results, cl, t);
			}

			for (cl = 0; cl < ncl; cl++) {
				*(double *)PyArray_GETPTR2(results, cl, t) /= denom;
			}
		}
	}
}

static PyObject* naivebayes(PyObject *dummy, PyObject *args) {
	PyObject *arg1=NULL, *arg3=NULL, *arg4=NULL, *arg5=NULL, *arg6=NULL;
	PyArrayObject *nps=NULL, *npcl=NULL, *npcomp=NULL, *npresults=NULL, *nplike=NULL;

	// Parse the input from NumPy
	if (!PyArg_ParseTuple(args, "OOOO!O!", &arg1, &arg3, &arg4,
		&PyArray_Type, &arg5, &PyArray_Type, &arg6)) return NULL;

	nps = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
	npcl = (PyArrayObject *)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_IN_ARRAY);
	npcomp = (PyArrayObject *)PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_IN_ARRAY);
	npresults = (PyArrayObject *)PyArray_FROM_OTF(arg5, NPY_DOUBLE, NPY_INOUT_ARRAY);
	nplike = (PyArrayObject *)PyArray_FROM_OTF(arg6, NPY_DOUBLE, NPY_INOUT_ARRAY);

	// Fail if necessary
	if (nps == NULL || npcomp == NULL ||
		npcl == NULL || npresults == NULL || nplike == NULL) goto fail;

    if (PyArray_NDIM(npcomp) != 2) {PySys_WriteStdout("ERROR: Comparison data has wrong number of dimensions\n"); goto fail;};
	if (PyArray_NDIM(nps) != 3) {PySys_WriteStdout("ERROR: Singles have wrong number of dimensions\n"); goto fail;};
	if (PyArray_NDIM(npcl) != 2) {PySys_WriteStdout("ERROR: Priors have wrong number of dimensions\n"); goto fail;};
	if (PyArray_ITEMSIZE(npcl) != 8) {PySys_WriteStdout("ERROR: Priors is not of type np.float64\n"); goto fail;};

	// Run
	nb_comparison(nps, npcomp, npcl, nplike, npresults);

	Py_DECREF(nps);
	Py_DECREF(npcl);
	Py_DECREF(npcomp);
	Py_DECREF(npresults);
	Py_DECREF(nplike);
    Py_INCREF(Py_None);
    return Py_None;

    // Account for failure by setting error and returning null
    fail:
	    Py_XDECREF(nps);
		Py_XDECREF(npcl);
		Py_XDECREF(npcomp);
		PyArray_XDECREF_ERR(npresults);
		PyArray_XDECREF_ERR(nplike);
		return NULL;
}*/

void aode_comparison(PyArrayObject *npsingles, PyArrayObject *npjoints, PyArrayObject *npcompare, PyArrayObject *priors,
                     PyArrayObject *likelihood, PyArrayObject *results) {
	int t, cl, kc, c;
	int ncl = PyArray_DIM(npjoints, 0), nc = PyArray_DIM(npjoints, 2), nkc = PyArray_DIM(npjoints, 1),
	    nt = PyArray_DIM(npcompare, 1);
	double denom, sum, product, compkc, compc;

	#pragma omp parallel shared(npsingles, npjoints, npcompare, priors, nc, nkc, nt, results) private(t, cl, kc, c,denom, sum, product, compkc, compc)
	{
		#pragma omp for schedule(dynamic, 100)
		// Iterate over all time points
		for (t = 0; t < nt; t++) {
		//for (t = 0; t < 100; t++) {
			// Divide by the denominator
			denom = 0.0;
			for (cl = 0; cl < ncl; cl++) {
				sum = 0.0;
				for (kc = 0; kc < nkc; kc++) {
				    // Prob spiking*probability of spiking during condition +
					compkc = *(double *) PyArray_GETPTR2(npcompare, kc, t);
					product = *(double *) PyArray_GETPTR3(npsingles, cl, kc, 0)*compkc +
						*(double *) PyArray_GETPTR3(npsingles, cl, kc, 1)*(1.0 - compkc);

					for (c = 0; c < nc; c++) {
    					compc = *(double *) PyArray_GETPTR2(npcompare, c, t);
						product *= *(double *) PyArray_GETPTR4(npjoints, cl, kc, c, 0)*compkc*compc +
							*(double *) PyArray_GETPTR4(npjoints, cl, kc, c, 1)*compkc*(1 - compc) +
							*(double *) PyArray_GETPTR4(npjoints, cl, kc, c, 2)*(1 - compkc)*compc +
							*(double *) PyArray_GETPTR4(npjoints, cl, kc, c, 3)*(1 - compkc)*(1 - compc);
					}

					sum += product;
				}

                *(double *)PyArray_GETPTR2(likelihood, cl, t) = sum;
				*(double *)PyArray_GETPTR2(results, cl, t) = *(double *) PyArray_GETPTR2(priors, cl, t)*sum;
                denom += *(double *)PyArray_GETPTR2(results, cl, t);
			}

			for (cl = 0; cl < ncl; cl++) {
				*(double *)PyArray_GETPTR2(results, cl, t) /= denom;
			}
		}
	}
}

static PyObject* aode(PyObject *dummy, PyObject *args) {
	PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL, *arg4=NULL, *arg5=NULL, *arg6=NULL;
	PyArrayObject *nps=NULL, *npj=NULL, *npcl=NULL, *npcomp=NULL, *npresults=NULL, *nplike=NULL;

	// Parse the input from NumPy
	import_array();
	if (!PyArg_ParseTuple(args, "OOOOO!O!", &arg1, &arg2, &arg3, &arg4,
		&PyArray_Type, &arg5, &PyArray_Type, &arg6)) return NULL;

	nps = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	npj = (PyArrayObject *)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	npcl = (PyArrayObject *)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	npcomp = (PyArrayObject *)PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	npresults = (PyArrayObject *)PyArray_FROM_OTF(arg5, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
	nplike = (PyArrayObject *)PyArray_FROM_OTF(arg6, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

	// Fail if necessary
	if (nps == NULL || npj == NULL || npcomp == NULL ||
		npcl == NULL || npresults == NULL || nplike == NULL) goto fail;

    if (PyArray_NDIM(npj) != 4) {PySys_WriteStdout("ERROR: Joints have wrong number of dimensions\n"); goto fail;};
	if (PyArray_NDIM(npcomp) != 2) {PySys_WriteStdout("ERROR: Comparison data has wrong number of dimensions\n"); goto fail;};
	if (PyArray_NDIM(nps) != 3) {PySys_WriteStdout("ERROR: Singles have wrong number of dimensions\n"); goto fail;};
	if (PyArray_NDIM(npcl) != 2) {PySys_WriteStdout("ERROR: Priors have wrong number of dimensions\n"); goto fail;};
	if (PyArray_ITEMSIZE(npcl) != 8) {PySys_WriteStdout("ERROR: Priors is not of type np.float64\n"); goto fail;};

	// Run
	aode_comparison(nps, npj, npcomp, npcl, nplike, npresults);

	Py_DECREF(nps);
	Py_DECREF(npj);
	Py_DECREF(npcl);
	Py_DECREF(npcomp);
	Py_DECREF(npresults);
	Py_DECREF(nplike);
    Py_INCREF(Py_None);
    return Py_None;

    // Account for failure by setting error and returning null
    fail:
	    Py_XDECREF(nps);
		Py_XDECREF(npj);
		Py_XDECREF(npcl);
		Py_XDECREF(npcomp);
		PyArray_XDECREF_ERR(npresults);
		PyArray_XDECREF_ERR(nplike);
		return NULL;
}

void runrollmax(PyArrayObject *npinput, PyArrayObject *npoutput, int integrate_frames) {
    /*
    ROLLMAX(input_array, output_array, integrate_frames
    */

    npy_intp i, j, c, s1, s2;

    // Designed to handle doubles
    if (PyArray_ITEMSIZE(npinput) != 8) npoutput = NULL;

    if (PyArray_NDIM(npinput) == 1) {
        s1 = PyArray_DIM(npoutput, 0);

        for (i = 0; i < s1; i++) {
            *(double *)PyArray_GETPTR1(npoutput, i) = *(double *)PyArray_GETPTR1(npinput, i);

            for (c = 1; c < integrate_frames; c++) {
                *(double *)PyArray_GETPTR1(npoutput, i) = *(double *)PyArray_GETPTR1(npinput, i+c) >
                    *(double *)PyArray_GETPTR1(npoutput, i) ? *(double *)PyArray_GETPTR1(npinput, i+c) :
                    *(double *)PyArray_GETPTR1(npoutput, i);
            }
        }
    }

    else if (PyArray_NDIM(npinput) == 2) {
        s1 = PyArray_DIM(npinput, 0);
        s2 = PyArray_DIM(npoutput, 1);

        for (i = 0; i < s1; i++) {
            for (j = 0; j < s2; j++) {
                *(npy_double *)PyArray_GETPTR2(npoutput, i, j) = *(npy_double *)PyArray_GETPTR2(npinput, i, j);

                for (c = 1; c < integrate_frames; c++) {
                    *(npy_double *)PyArray_GETPTR2(npoutput, i, j) = *(npy_double *)PyArray_GETPTR2(npinput, i, j+c) >
                        *(npy_double *)PyArray_GETPTR2(npoutput, i, j) ? *(npy_double *)PyArray_GETPTR2(npinput, i, j+c) :
                        *(npy_double *)PyArray_GETPTR2(npoutput, i, j);
                }
            }
        }
    }

    else {
        npoutput = NULL;
    }
}

static PyObject* rollmax(PyObject *dummy, PyObject *args) {
	int integrate_frames;
	PyObject *arg1=NULL, *arg2=NULL;  // inputs will be the input array and the output array
	PyArrayObject *npinput=NULL, *npoutput=NULL; // input array and output array

    // Parse the input from NumPy
	import_array();
	if (!PyArg_ParseTuple(args, "OOi", &arg1, &arg2, &integrate_frames)) return NULL;
    npinput = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	npoutput = (PyArrayObject *)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    // Fail if necessary
	if (npinput == NULL || npoutput == NULL) goto fail;

    // Run
    runrollmax(npinput, npoutput, integrate_frames);

	if (npinput == NULL) {PySys_WriteStdout("ERROR: Problems receiving input\n");}
	if (npoutput == NULL) {PySys_WriteStdout("ERROR: Problems in running rolling max\n");}

	Py_DECREF(npinput);
	Py_DECREF(npoutput);
    Py_INCREF(Py_None);
    return Py_None;

    // Account for failure by setting error and returning null
    fail:
	    Py_XDECREF(npinput);
		PyArray_XDECREF_ERR(npoutput);
		return NULL;
}

void runrollmean(PyArrayObject *npinput, PyArrayObject *npoutput, int integrate_frames) {
    /*
    ROLLMEAN(input_array, output_array, integrate_frames
    */

    npy_intp i, j, c, s1, s2;

    // Designed to handle doubles
    if (PyArray_ITEMSIZE(npinput) != 8) npoutput = NULL;

    if (PyArray_NDIM(npinput) == 1) {
        s1 = PyArray_DIM(npoutput, 0);

        for (i = 0; i < s1; i++) {
            *(double *)PyArray_GETPTR1(npoutput, i) = *(double *)PyArray_GETPTR1(npinput, i)/integrate_frames;

            for (c = 1; c < integrate_frames; c++) {
                *(double *)PyArray_GETPTR1(npoutput, i) = *(double *)PyArray_GETPTR1(npoutput, i) +
                                                          *(double *)PyArray_GETPTR1(npinput, i+c)/integrate_frames;
            }
        }
    }

    else if (PyArray_NDIM(npinput) == 2) {
        s1 = PyArray_DIM(npinput, 0);
        s2 = PyArray_DIM(npoutput, 1);

        for (i = 0; i < s1; i++) {
            for (j = 0; j < s2; j++) {
                *(npy_double *)PyArray_GETPTR2(npoutput, i, j) =
                    *(npy_double *)PyArray_GETPTR2(npinput, i, j)/integrate_frames;

                for (c = 1; c < integrate_frames; c++) {
                    *(npy_double *)PyArray_GETPTR2(npoutput, i, j) = *(npy_double *)PyArray_GETPTR2(npoutput, i, j) +
                        *(npy_double *)PyArray_GETPTR2(npinput, i, j+c)/integrate_frames;
                }
            }
        }
    }

    else {
        npoutput = NULL;
    }
}

static PyObject* rollmean(PyObject *dummy, PyObject *args) {
	int integrate_frames;
	PyObject *arg1=NULL, *arg2=NULL;  // inputs will be the input array and the output array
	PyArrayObject *npinput=NULL, *npoutput=NULL; // input array and output array

	// Parse the input from NumPy
	import_array();
    if (!PyArg_ParseTuple(args, "OOi", &arg1, &arg2, &integrate_frames)) return NULL;

	npinput = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	npoutput = (PyArrayObject *)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

	// Fail if necessary
	if (npinput == NULL || npoutput == NULL) goto fail;

    // Run
    runrollmean(npinput, npoutput, integrate_frames);

	if (npinput == NULL) {PySys_WriteStdout("ERROR: Problems receiving input\n");}
	if (npoutput == NULL) {PySys_WriteStdout("ERROR: Problems in running rolling mean\n");}

	Py_DECREF(npinput);
	Py_DECREF(npoutput);
    Py_INCREF(Py_None);
    return Py_None;

    // Account for failure by setting error and returning null
    fail:
	    Py_XDECREF(npinput);
		PyArray_XDECREF_ERR(npoutput);
		return NULL;
}

static struct PyMethodDef methods[] = {
    {"aode", aode, METH_VARARGS, "Takes two large arrays and computes AODE"},
    {"naivebayes", naivebayes, METH_VARARGS, "Takes two large arrays and computes Naive Bayes"},
    {"rollmax", rollmax, METH_VARARGS, "Computes a rolling maximum across the last axis of an array"},
    {"rollmean", rollmean, METH_VARARGS, "Computes a rolling mean across the last axis of an array"},
    {NULL, NULL, 0, NULL}
};

#ifdef PY3K
static struct PyModuleDef runclassifier = {
    PyModuleDef_HEAD_INIT,
    "runclassifier", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    methods
};

PyMODINIT_FUNC PyInit_runclassifier(void) {
    return PyModule_Create(&runclassifier);
};
#else
PyMODINIT_FUNC
initrunclassifier (void) {
    (void) Py_InitModule("runclassifier", methods);
    import_array();
}
#endif
