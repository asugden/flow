#include "Python.h"
#include "numpy/arrayobject.h"
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

typedef struct {
	int nclasses;
	int nkeycells;
	double *data;
} SProbs;

typedef struct {
	int ncells;
	int ntimes;
	double *data;
} BinData;

typedef struct {
	int nclasses;
	int ntimes;
	double *data;
} Results;

#define sp(u, cl, kc, tf, nkc) (u->data[((nkc)*2)*(cl) + 2*(kc) + (tf)])
#define bdp(u, c, t, nt) (u->data[(c)*(nt) + (t)])
#define PSEUDOCOUNT 0.1


/* Compile with cc tracing_components/aode.c -I 
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/
 -I /opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7/
 */

SProbs *getsingles(PyArrayObject *npsingles) {
	int cl, kc;
	SProbs *singles = malloc(sizeof(SProbs));

	if (PyArray_NDIM(npsingles) != 3) return NULL;
	singles->nclasses = PyArray_DIM(npsingles, 0);
	singles->nkeycells = PyArray_DIM(npsingles, 1);

	singles->data = malloc(singles->nclasses*singles->nkeycells*2*sizeof(double));

	for (cl = 0; cl < singles->nclasses; cl++) {
		for (kc = 0; kc < singles->nkeycells; kc++) {
			sp(singles, cl, kc, 0, singles->nkeycells) = 
				*(double *) PyArray_GETPTR3(npsingles, cl, kc, 0);
			sp(singles, cl, kc, 1, singles->nkeycells) = 
				*(double *) PyArray_GETPTR3(npsingles, cl, kc, 1);
		}
	}

	return singles;
}

BinData *getbindata(PyArrayObject *compdata, int integrate_frames) {
	int c, t, i, ncells, ntimes;
	BinData *compare = malloc(sizeof(BinData));

	if (PyArray_NDIM(compdata) != 2) return NULL;
	if (PyArray_ITEMSIZE(compdata) != 8) return NULL;
	ncells = PyArray_DIM(compdata, 0);
	ntimes = PyArray_DIM(compdata, 1) - (integrate_frames - 1);

	compare->ncells = ncells;
	compare->ntimes = ntimes - integrate_frames + 1;
	compare->data = malloc(ncells*(ntimes - integrate_frames + 1)*sizeof(double));

	for (c = 0; c < compare->ncells; c++) {
		for (t = 0; t < compare->ntimes; t++) {
			bdp(compare, c, t, compare->ntimes) = *(double *) PyArray_GETPTR2(compdata, c, t);
			for (i = 1; i < integrate_frames; i++) {
				bdp(compare, c, t, compare->ntimes) = (bdp(compare, c, t, compare->ntimes) > 
					*(double *) PyArray_GETPTR2(compdata, c, t+i) ? bdp(compare, c, t, compare->ntimes) : 
					*(double *) PyArray_GETPTR2(compdata, c, t+i));
			}
		}
	}

	return compare;
}

Results *getpriors(PyArrayObject *npcl) {
	int cl, nt, t, nclasses;
	Results *priors = malloc(sizeof(Results));

	if (PyArray_NDIM(npcl) != 2) return NULL;
	if (PyArray_ITEMSIZE(npcl) != 8) return NULL;
	nclasses = PyArray_DIM(npcl, 0);
	nt = PyArray_DIM(npcl, 1);

	priors->nclasses = nclasses;
	priors->ntimes = nt;
	priors->data = malloc(nclasses*nt*sizeof(double));

	for (cl = 0; cl < priors->nclasses; cl++) {
		for (t = 0; t < priors->ntimes; t++) {
		    priors->data[cl*nt + t] = *(double *) PyArray_GETPTR2(npcl, cl, t);
		}
	}

	return priors;
}

double *getclassprobabilities(PyArrayObject *npcl, SProbs *singles, int add_baseline) {
	/* Convert a numpy array of class probabilities to an array of
	doubles. If baseline is to be added, it MUST have been sent by 
	Python. */

	int cl;
	int nclasses = PyArray_DIM(npcl, 0);

	// Account for baseline
	if (add_baseline == 0 && singles->nclasses != nclasses) {
		PySys_WriteStdout("ERROR: Number of classes does not match between singles (%i) and priors (%i)\n", 
			singles->nclasses, nclasses);
		return NULL;
	}
	if (add_baseline > 0 && singles->nclasses != nclasses) {
		PySys_WriteStdout("ERROR: add-baseline, Number of classes does not match between singles (%i) and priors (%i)\n", 
			singles->nclasses, nclasses);
		return NULL;
	}

	double *classprobs = malloc(nclasses*sizeof(double));

	for (cl = 0; cl < nclasses; cl++) {
		classprobs[cl] = *(double *)PyArray_GETPTR1(npcl, cl);
	}

	return classprobs;
}

void getbaseline_partial(SProbs *singles, PyArrayObject *compdata) {
	/* Make a new baseline class that has probabilities set to the 
	activity levels of the data in the comparison run. */
	int t, kc, tf, cl = singles->nclasses - 1, nkc = singles->nkeycells;
	int ntimes = PyArray_DIM(compdata, 1);
	double denom;
	
	#pragma omp parallel shared(singles, compdata, cl, nkc, ntimes) private(t, kc, tf, denom)
	{
		#pragma omp for schedule(dynamic, 100)
		// Add up number of single and joint events
		for (t = 0; t < ntimes; t++) {
			for (kc = 0; kc < nkc; kc++) {
				if (t == 0) {
					sp(singles, cl, kc, 0, nkc) = 2*PSEUDOCOUNT;
					sp(singles, cl, kc, 1, nkc) = 2*PSEUDOCOUNT;
				}

				tf = *(double *) PyArray_GETPTR2(compdata, kc, t) > 0 ? 0 : 1;
				sp(singles, cl, kc, tf, nkc)++; // 0:T, 1:F
			}
		}
	}

	// Convert to probabilities
	for (kc = 0; kc < nkc; kc++) {
		denom = sp(singles, cl, kc, 0, nkc) + sp(singles, cl, kc, 1, nkc);
		sp(singles, cl, kc, 0, nkc) /= denom;
		sp(singles, cl, kc, 1, nkc) /= denom;
	}
}

void setresults(PyArrayObject *results, double *out, int nintegratedtimes) {
	/* Copy over the results array to the appropriate numpy object and
	free the resulting memory. */
	int cl, t;
	int nclasses = PyArray_DIM(results, 0);

	// Iterate over the points, copying the values
	for (cl = 0; cl < nclasses; cl++) {
		for (t = 0; t < nintegratedtimes; t++) {
			*(double *)PyArray_GETPTR2(results, cl, t) = out[cl*nintegratedtimes + t];
		}
	}

	// Clean up after oneself
	free(out);
}

void setbaseline(PyArrayObject *npsingles, SProbs *singles) {
	/* If the baseline data was demonstrated, copy it over to the numpy 
	arrays so that the data can be reviewed back in Python. */

	int kc, cl = singles->nclasses-1, nkc = singles->nkeycells;

	// Convert to probabilities
	for (kc = 0; kc < singles->nkeycells; kc++) {
		*(double *) PyArray_GETPTR3(npsingles, cl, kc, 0) = sp(singles, cl, kc, 0, nkc);
		*(double *) PyArray_GETPTR3(npsingles, cl, kc, 1) = sp(singles, cl, kc, 1, nkc);
	}
}

double *nb_comparison(SProbs *singles, BinData *compare, Results *priors, PyArrayObject *likelihood) {
	int t, cl, kc;
	int nkc = singles->nkeycells, nt = compare->ntimes;
	double denom, product;
	double *results = malloc(singles->nclasses*compare->ntimes*sizeof(double));

	#pragma omp parallel shared(singles, compare, priors, nkc, nt, results) private(t, cl, kc, denom, product)
	{
		#pragma omp for schedule(dynamic, 100)
		// Iterate over all time points
		for (t = 0; t < compare->ntimes; t++) {
		//for (t = 0; t < 100; t++) {
			// Divide by the denominator
			denom = 0.0;
			for (cl = 0; cl < singles->nclasses; cl++) {
				product = 1.0;
				for (kc = 0; kc < singles->nkeycells; kc++) {
					// Prob spiking*probability of spiking during condition + 
					product *= bdp(compare, kc, t, nt)*sp(singles, cl, kc, 0, nkc) + 
						(1.0 - bdp(compare, kc, t, nt))*sp(singles, cl, kc, 1, nkc);
				}

                *(double *)PyArray_GETPTR2(likelihood, cl, t) = product;
				results[cl*nt + t] = product*priors->data[cl*nt + t];
				denom += results[cl*nt + t];
			}

			for (cl = 0; cl < singles->nclasses; cl++) {
				results[cl*nt + t] /= denom;
			}
		}
	}

	return results;
}

/*
static PyObject* train(PyObject *dummy, PyObject *args) {

}
*/

static PyObject* compare(PyObject *dummy, PyObject *args) {
	int integrate_frames=0, add_baseline=0;
	PyObject *arg1=NULL, *arg3=NULL, *arg4=NULL, *arg5=NULL, *arg6=NULL;
	PyArrayObject *nps=NULL, *npcl=NULL, *npcomp=NULL, *npresults=NULL, *nplike=NULL;

	// Parse the input from NumPy
	if (!PyArg_ParseTuple(args, "OOOiiO!O!", &arg1, &arg3, &arg4,
		&integrate_frames, &add_baseline, &PyArray_Type, &arg5, &PyArray_Type, &arg6)) return NULL;

	nps = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
	npcl = (PyArrayObject *)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_IN_ARRAY);
	npcomp = (PyArrayObject *)PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_IN_ARRAY);
	npresults = (PyArrayObject *)PyArray_FROM_OTF(arg5, NPY_DOUBLE, NPY_INOUT_ARRAY);
	nplike = (PyArrayObject *)PyArray_FROM_OTF(arg6, NPY_DOUBLE, NPY_INOUT_ARRAY);

	// Fail if necessary
	if (nps == NULL || npcomp == NULL ||
		npcl == NULL || npresults == NULL || nplike == NULL) goto fail;

	// Copy to useable arrays
	SProbs *singles = getsingles(nps);
	BinData *compare = getbindata(npcomp, integrate_frames);
	// Get prior probabilities and double-check that array sizes are
	// appropriate
	// double *priors = getclassprobabilities(npcl, singles, joints, add_baseline);
	Results *priors = getpriors(npcl);
	if (singles == NULL) {PySys_WriteStdout("ERROR: Singles\n");}
	if (compare == NULL) {PySys_WriteStdout("ERROR: Comparison data\n");}
	if (priors == NULL) {PySys_WriteStdout("ERROR: Priors\n");}
	if (singles == NULL || compare == NULL || priors == NULL) goto fail;
	
	if (add_baseline > 0) {getbaseline_partial(singles, npcomp);}
	// Run
	/*int i;
	for (i = 0; i < 100; i++) {
		//PySys_WriteStdout("%.10e\n", *(double *) PyArray_GETPTR2(npcomp, i, 2));
		//PySys_WriteStdout("%i\n", bdp(compare, i, 2, compare->ntimes));
	}*/
	double *res = nb_comparison(singles, compare, priors, nplike);
	
	// Set the output
	setresults(npresults, res, compare->ntimes);
	if (add_baseline > 0) {setbaseline(nps, singles);}
	
	// Clean up after oneself
	free(singles->data);
	free(singles);
	free(compare->data);
	free(compare);
	free(priors);

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

static struct PyMethodDef methods[] = {
    {"compare", compare, METH_VARARGS, "Takes two large arrays and computes NaiveBayes"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initanb (void) {
    (void) Py_InitModule("anb", methods);
    import_array();
}
