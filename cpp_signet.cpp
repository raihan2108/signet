#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
using namespace std;
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75

const int hash_table_size = 30000000;
const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;


typedef float real;

struct ClassVertex {
	double degree;
	long name;
};

// char network_file[MAX_STRING], embedding_file[MAX_STRING];
struct ClassVertex *vertex;
int is_binary = 0, num_threads = 1, order = 2, dim = 100, num_negative = 5;
int *vertex_hash_table, *neg_table;
int max_num_vertices = 1000, num_vertices = 0;
long long total_samples = 1, current_sample_count = 0, num_edges = 0;
real init_rho = 0.025, rho;
real *emb_vertex, *emb_context, *sigmoid_table;

long *edge_source_id, *edge_target_id;
double *edge_weight;

map<long, vector<long> > pos_sample;
map<long, vector<long> > neg_sample;

// Parameters for edge sampling
long long *alias;
double *prob;
bool do_neg_sampling;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;


long long SampleAnEdge(double rand_value1, double rand_value2)
{
	long long k = (long long)num_edges * rand_value1;
	return rand_value2 < prob[k] ? k : alias[k];
}


void InitAliasTable()
{
	alias = (long long *)malloc(num_edges*sizeof(long long));
	prob = (double *)malloc(num_edges*sizeof(double));
	if (alias == NULL || prob == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double *norm_prob = (double*)malloc(num_edges*sizeof(double));
	long long *large_block = (long long*)malloc(num_edges*sizeof(long long));
	long long *small_block = (long long*)malloc(num_edges*sizeof(long long));
	if (norm_prob == NULL || large_block == NULL || small_block == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double sum = 0;
	long long cur_small_block, cur_large_block;
	long long num_small_block = 0, num_large_block = 0;

	for (long long k = 0; k != num_edges; k++) sum += abs(edge_weight[k]);
	for (long long k = 0; k != num_edges; k++) norm_prob[k] = abs(edge_weight[k]) * num_edges / sum;

	for (long long k = num_edges - 1; k >= 0; k--)
	{
		if (norm_prob[k]<1)
			small_block[num_small_block++] = k;
		else
			large_block[num_large_block++] = k;
	}

	while (num_small_block && num_large_block)
	{
		cur_small_block = small_block[--num_small_block];
		cur_large_block = large_block[--num_large_block];
		prob[cur_small_block] = norm_prob[cur_small_block];
		alias[cur_small_block] = cur_large_block;
		norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
		if (norm_prob[cur_large_block] < 1)
			small_block[num_small_block++] = cur_large_block;
		else
			large_block[num_large_block++] = cur_large_block;
	}

	while (num_large_block) prob[large_block[--num_large_block]] = 1;
	while (num_small_block) prob[small_block[--num_small_block]] = 1;

	free(norm_prob);
	free(small_block);
	free(large_block);
}


void InitNegTable()
{
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	neg_table = (int *)malloc(neg_table_size * sizeof(int));
	for (int k = 0; k != num_vertices; k++) sum += pow(vertex[k].degree, NEG_SAMPLING_POWER);
	for (int k = 0; k != neg_table_size; k++)
	{
		if ((double)(k + 1) / neg_table_size > por)
		{
			cur_sum += pow(vertex[vid].degree, NEG_SAMPLING_POWER);
			por = cur_sum / sum;
			vid++;
		}
		neg_table[k] = vid - 1;
	}
}


void InitSigmoidTable()
{
	real x;
	sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
	// sigmoid_table = new real[sigmoid_table_size + 1];
	for (int k = 0; k != sigmoid_table_size; k++)
	{
		x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
		sigmoid_table[k] = 1 / (1 + exp(-x));
	}
}


real FastSigmoid(real x)
{
	if (x > SIGMOID_BOUND) return 1;
	else if (x < -SIGMOID_BOUND) return 0;
	int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
	return sigmoid_table[k];
}


int Rand(unsigned long long &seed)
{
	seed = seed * 25214903917 + 11;
	return (seed >> 16) % neg_table_size;
}


int initVertex(long n_vertices)
{
    for(int i = 0; i < n_vertices; i++){
        vertex[i].name = i;
	    vertex[i].degree = 0;
    }

	return 0;
}


long sample_pos_edge(long node, unsigned long long seed)
{
    return pos_sample[node][Rand(seed) % pos_sample[node].size()];
}


long sample_neg_edge(long node, unsigned long long seed)
{
    return neg_sample[node][Rand(seed) % neg_sample[node].size()];
}


void Update(real *vec_u, real *vec_v, real *vec_error, int label)
{
	real x = 0, g;
	for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
	g = (label - FastSigmoid(x)) * rho;
	for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
	for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
}


void *TrainSEINEThread(void *id)
{
    long long u, v, w, lu, lv, target, label;
    long long count = 0, last_count = 0, curedge;
	unsigned long long seed = (long long)id;
	real *vec_error = (real *)calloc(dim, sizeof(real));
	while (1)
	{
		//judge for exit
		if (count > total_samples / num_threads + 2) break;
		if (count - last_count > 10000)
		{
			current_sample_count += count - last_count;
			last_count = count;
			printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)(total_samples + 1) * 100);
			fflush(stdout);
			rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
        }
        curedge = SampleAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
		u = edge_source_id[curedge];
		v = edge_target_id[curedge];
		w = edge_weight[curedge];

		lu = u * dim;
		for (int c = 0; c != dim; c++) vec_error[c] = 0;

		for (int d = 0; d <= num_negative; d++)
		{
		    if (d == 0)
		    {
		        label =  w > 0 ? 1 : 0;
		        target = v;
		    }
		    else if (do_neg_sampling)
		    {
                target = neg_table[Rand(seed)];
				label = 0;
		    }
		    else
		    {
		        if (pos_sample[u].size() < num_negative || neg_sample[u].size()  < num_negative) break;
		        label =  w > 0 ? 0 : 1;
                target = w > 0 ? sample_neg_edge(u, seed) : sample_pos_edge(u, seed);
		    }
		    lv = target * dim;
		    if (order == 1) Update(&emb_vertex[lu], &emb_vertex[lv], vec_error, label);
			if (order == 2) Update(&emb_vertex[lu], &emb_context[lv], vec_error, label);

		}
		for (int c = 0; c != dim; c++) emb_vertex[c + lu] += vec_error[c];

		count++;
    }

}


void TrainSEINE() {
    long a;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

	gsl_rng_env_setup();
	gsl_T = gsl_rng_default;
	gsl_r = gsl_rng_alloc(gsl_T);
	gsl_rng_set(gsl_r, 314159265);

	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainSEINEThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

}


void signet(long* source, long* target, long* weight, double* embed_vertex, double* embed_context,
          map<long, vector<long> > samp, map<long, vector<long> > samn,
          long n_vertices, long n_edges, long n_dims, double init_rho, long n_iterations,
          long n_negatives, long n_order, long n_samples, bool is_sample)
{
    cout << "entering signet method" << endl;

    long long a, b;
    edge_source_id = new long[n_edges];
    edge_target_id = new long[n_edges];
    edge_weight = new double[n_edges];
    vertex = new ClassVertex[n_vertices];

    num_negative = n_negatives;
    dim = n_dims;
    order = n_order;
    total_samples = n_samples;
    total_samples *= 1000000;
	rho = init_rho;
	num_edges = n_edges;
	num_vertices = n_vertices;
	do_neg_sampling = is_sample;

	initVertex(n_vertices);

    for(int i = 0; i < n_edges; i++)
    {
        edge_source_id[i] = source[i];
        edge_target_id[i] = target[i];
        edge_weight[i] = weight[i];
        vertex[edge_source_id[i]].degree += abs(edge_weight[i]);
    }

    ;
    for (map<long, vector<long> >::iterator it = samp.begin(); it != samp.end(); it ++)
    {
        long curr_node = it->first;
        vector<long> pos_entry = vector<long>();
        for(int j = 0; j != samp[curr_node].size(); j ++)
        {
            pos_entry.push_back(samp[curr_node][j]);
        }
        pos_sample[curr_node] = pos_entry;

        vector<long> neg_entry = vector<long>();
        for(int j = 0; j != samn[curr_node].size(); j ++)
        {
            neg_entry.push_back(samn[curr_node][j]);
        }
        neg_sample[curr_node] = neg_entry;

    }
    if (do_neg_sampling)
        InitNegTable();
	InitSigmoidTable();
	InitAliasTable();

	emb_vertex = new real[n_vertices * n_dims];
    emb_context = new real[n_vertices * n_dims];
    for (b = 0; b < dim; b++)
        for (a = 0; a < num_vertices; a++)
		    emb_vertex[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
    for (b = 0; b < dim; b++)
        for (a = 0; a < num_vertices; a++)
		    emb_context[a * dim + b] = 0;

    clock_t start = clock();

    TrainSEINE();

    clock_t finish = clock();

    for(a =0; a < num_vertices; a++)
    {
        double len = 0.0;
        for(b= 0; b < n_dims; b++) len += emb_vertex[a * n_dims + b] * emb_vertex[a * n_dims + b];
        len = sqrt(len);
        for(b= 0; b < n_dims; b++) emb_vertex[a * n_dims + b] /= len;

        len = 0.0;
        for(b= 0; b < n_dims; b++) len += emb_context[a * n_dims + b] * emb_context[a * n_dims + b];
        len = sqrt(len);
        if (len != 0)
            for(b= 0; b < n_dims; b++) emb_context[a * n_dims + b] /= len;
    }

    for (b = 0; b < n_dims; b++)
        for (a = 0; a < num_vertices; a++)
		    embed_vertex[a * n_dims + b] = emb_vertex[a * n_dims + b];
    for (b = 0; b < n_dims; b++)
        for (a = 0; a < num_vertices; a++)
		    embed_context[a * n_dims + b] = emb_context[a * n_dims + b];

	printf("\nTotal time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

    delete[] edge_source_id;
    delete[] edge_target_id;
    delete[] edge_weight;
    delete[] vertex;
    delete[] emb_context;
    delete[] emb_vertex;

    gsl_rng_free(gsl_r);
}



