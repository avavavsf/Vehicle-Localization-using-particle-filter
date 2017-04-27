/*
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 * Finalized: April 26, 2017 Yunming Shao 
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include "particle_filter.h"
#include "helper_functions.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
    
    num_particles = 80;

	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	for(int i=0;i < num_particles; i++){
		Particle particle;
		//particle.id = i;
		particle.x = dist_x(gen);
		//std::cout << particle.x <<std::endl;
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
		weights.push_back(particle.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	// http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	// http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0 , std_pos[2]);

	for(int i=0; i< num_particles;i++){
		Particle &particle = particles[i];
		double new_theta = particle.theta + yaw_rate * delta_t;
    // this could be a problem, we assume yaw_rate is no zero
		particle.x = particle.x + (velocity/yaw_rate) * (sin(new_theta) - sin(particle.theta));
		particle.y = particle.y + (velocity/yaw_rate) * (cos(particle.theta) -  cos(new_theta));
		particle.theta = new_theta;

    //add noise
		particle.x += dist_x(gen);
		particle.y += dist_y(gen);
		particle.theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> landmarks_in_map_coord, std::vector<LandmarkObs>& observations_in_map_coord) {
	//Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// iterate through observations
	// assume the observation are already converted to map coordinates system
	for (auto & obs : observations_in_map_coord) {
    	obs.id = -1;
    	double min_dist_sq = 999999;

    	// check distance to every landmark, map to nearest
    	for (auto & pred : landmarks_in_map_coord) {
          double dist_sq = dist(obs.x, obs.y, pred.x, pred.y);
      		if (min_dist_sq > dist_sq) {
        		min_dist_sq = dist_sq;
        		obs.id = pred.id;
      		}
    	}
  	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
  	std::vector<LandmarkObs> observations, Map map_landmarks) {
  	//Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  	//NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
  	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  	//   The following is a good resource for the theory:
  	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
 	//   and the following is a good resource for the actual equation to implement (look at equation 
  	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
  	//   for the fact that the map's y-axis actually points downwards.)
  	//   http://planning.cs.uiuc.edu/node99.html

	   // some useful parameters
    double two_sigma_x_sq = 2 * std_landmark[0] * std_landmark[0];
    double two_sigma_y_sq = 2 * std_landmark[1] * std_landmark[1];
  	double one_over_two_pi_sigma_sq = 1.0 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

  	weights.clear();
  	for (int i = 0; i < num_particles; ++i) {
    	Particle p = particles[i];

    	// Convert meaurement to map coordinates based on particles position
    	double cost = cos(p.theta);
    	double sint = sin(p.theta);
    	std::vector<LandmarkObs> observations_tx;
    	for (int j = 0; j < observations.size(); j++) {
      		// Translate to map coordinates
      		LandmarkObs obs_m;
      		obs_m.x = (observations[j].x * cost - observations[j].y * sint) + p.x;
      		obs_m.y = (observations[j].x * sint + observations[j].y * cost) + p.y;
      		observations_tx.push_back(obs_m);
    	}

    	// Select landmarks that are in range
    	std::vector<LandmarkObs> mapLandmarks;
    	for (auto &landmark : map_landmarks.landmark_list){
          if (dist(p.x, p.y, landmark.x_f, landmark.y_f) <= sensor_range) {
        		LandmarkObs l;
        		l.id = landmark.id_i;
        		l.x = landmark.x_f;
        		l.y = landmark.y_f;
        		mapLandmarks.push_back(l);
      		}
    	}

    	// Find nearest landmarks for each measurement
    	dataAssociation(mapLandmarks, observations_tx);

    	// Compute weight
    	p.weight = 1;
    	for (auto& obs : observations_tx) {
      		if (obs.id != -1 && p.weight > 0) {
        		//if (map_landmarks.landmark_list[obs.id - 1].id_i != obs.id)
         		// throw new exception("unexpected map item");
        		double delta_x = obs.x - map_landmarks.landmark_list[obs.id - 1].x_f;
        		double delta_y = obs.y - map_landmarks.landmark_list[obs.id - 1].y_f;
        		double prob = one_over_two_pi_sigma_sq * exp(-(((delta_x*delta_x) / two_sigma_x_sq) + ((delta_y*delta_y) / two_sigma_y_sq)));
        		p.weight *= prob;
      		}
    	}
    	weights.push_back(p.weight);
  	}
}

void ParticleFilter::resample() {
  //Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::default_random_engine gen;
  	std::discrete_distribution<int> dist(weights.begin(), weights.end());
  	std::vector<Particle> new_particles;
  	// pick particles with replacement
  	for (int i = 0; i < num_particles; i++) {
    	new_particles.push_back(particles[dist(gen)]);
  	}
  	particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
