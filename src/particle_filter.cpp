/*
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 * Finalized: April 16, 2017 Yunming Shao 
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
    
    num_particles = 80;
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std_x);
	std::normal_distribution<double> dist_y(y, std_y);
	std::normal_distribution<double> dist_theta(theta, std_theta);

	for(int i=0;i < num_particles; i++){
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
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

	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	for(int i=0; i< num_particles;i++){
		Particle &particle = particles[i];
		double new_theta = particle.theta + yaw_rate * delta_t;
		particle.x = particle.x + (velocity/yaw_rate) * (sin(new_theta) - sin(particle.theta));
		particle.y = particle.y + (velocity/yaw_rate) * (cos(particle.theta) -  cos(new_theta));
		particle.theta = new_theta;

		std::normal_distribution<double> dist_x(particle.x, std_x);
		std::normal_distribution<double> dist_y(particle.y, std_y);
		std::normal_distribution<double> dist_theta(particle.theta , std_theta);

		particle.id = i; 
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
	}
}

void transform_landmarks_coord(std::vector<LandmarkObs>& transformed_landmarks, const Particle& particle, const Map& map_landmarks) {
/**
 * Transform the map landmarks to the particle coordinate system.
 * https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/2c318113-724b-4f9f-860c-cb334e6e4ad7/lessons/5c50790c-5370-4c80-aff6-334659d5c0d9/concepts/1da01dfe-6653-4d2d-b2e4-325740fc7156
 * @param particle: The particle whose coordinate system defines the transformation
 * @param map_landmarks: Map class containing map landmarks
 * @output transformed_landmarks: The vector of landmarks transformed to the particle coordinate system
 */
	
	for (int i = 0; i < map_landmarks.landmark_list.size(); i++) {
		const Map::single_landmark_s& landmark = map_landmarks.landmark_list[i];
		LandmarkObs transformed_landmark;
		transformed_landmark.id = landmark.id_i;
		double cos_theta = cos(particle.theta - M_PI / 2);
		double sin_theta = sin(particle.theta - M_PI / 2);
		transformed_landmark.x = -(landmark.x_f - particle.x) * sin_theta + (landmark.y_f - particle.y) * cos_theta;
		transformed_landmark.y = -(landmark.x_f - particle.x) * cos_theta - (landmark.y_f - particle.y) * sin_theta;
		transformed_landmarks.push_back(transformed_landmark);
	}

}

// x_map and y_map will have x and y translated to map coordinates
void CarToMap(double x, double y, double x_car_map, double y_car_map, double phi_car_map, double& x_map, double & y_map){
  // using http://planning.cs.uiuc.edu/node99.html
  // Assuming map x axis points right, y axis points up.
  // Assuming car x axis points forward, y axis points left.
  x_map = x * cos(phi_car_map) - y * sin(phi_car_map) + x_car_map;
  y_map = x * sin(phi_car_map) + y * cos(phi_car_map) + y_car_map;
}

void data_association_per_particle(std::vector<LandmarkObs>& predicteds, const std::vector<LandmarkObs>& observations,
		const Map& map_landmarks, const Particle& particle){
	/**
	* Associate each observation to its mostly likely predicted landmark measurements for a particular particle
	* using Nearest Neighbor methods
	* @param observations, the list of actual landmark measurements
    * @param map_landmarks, all available landmarks in the map
    * @param particle, the particle being processed
    * @output predicteds, the vector of predicted landmark measurements
    */
	for (int i =0; i< observations.size();i++){
		const LandmarkObs& landmarkobs = observations[i];
		//std::cout <<observations.size() <<std::endl;
		//std::cout <<landmarkobs.x <<std::endl;

		//transform all landmarks in map to particle coordinate system
		//std::vector<LandmarkObs> transformed_landmarks;
		//transform_landmarks_coord(transformed_landmarks, particle, map_landmarks);

		//std::vector<LandmarkObs> observations_map_coord; // observations in map coordinates
		//CarToMap(landmarkobs.x, landmarkobs.y, particle.x, particle.y, particle.theta, landmarkobs.x, landmarkobs.y); // we are overwriting obs.x and obs.y inside function
        
        // using http://planning.cs.uiuc.edu/node99.html
  		// Assuming map x axis points right, y axis points up.
  		// Assuming car x axis points forward, y axis points left.
  		double landmarkobsInMapx = landmarkobs.x * cos(particle.theta) - landmarkobs.y * sin(particle.theta) + particle.x;
  		double landmarkobsInMapy = landmarkobs.x * sin(particle.theta) + landmarkobs.y * cos(particle.theta) + particle.y;
        //observations_map_coord.push_back(obs); // copies obs as new object

		//Find closet landmark as the predicted landmark
		double clostest_dist = -1;
		int predicted_landmark_ind = -1;
		for(int j=0; j< map_landmarks.landmark_list.size(); j++){
			//const LandmarkObs& map_landmark = map_landmarks[j];

			const Map::single_landmark_s& map_landmark = map_landmarks.landmark_list[j];

			double x_dist = map_landmark.x_f - landmarkobsInMapx;
			double y_dist = map_landmark.y_f - landmarkobsInMapy;
			double dist = sqrt(x_dist*x_dist + y_dist*y_dist);
			//sensor range information is not used to filer out those landmarks that are outside the sensor range for this particle.
			//This is because those distant landmarks, if matched with an observation,  will end up with predicted landmark measurements that is very different
			//from actual landmark measurements, and thus this particle can be effectively filtered due to  this mismatch

			if(clostest_dist == -1 || dist < clostest_dist ){
				clostest_dist = dist;
				predicted_landmark_ind = j;
			}
		}
		//transform predicted landmark to vehicle coordinate system
		LandmarkObs landmark_closet;
		landmark_closet.id = predicted_landmark_ind;
		landmark_closet.x = map_landmarks.landmark_list[predicted_landmark_ind].x_f;
		landmark_closet.y = map_landmarks.landmark_list[predicted_landmark_ind].y_f;

		std::cout << predicted_landmark_ind <<std::endl;
		std::cout << landmark_closet.id << ' '<< landmark_closet.x<<' '<< landmark_closet.y << std::endl;


		predicteds.push_back(landmark_closet);
	}

}

// The function is not used.
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	for(int i=0; i< num_particles;i++){
		Particle &particle = particles[i];
		std::vector<LandmarkObs> predicteds;

		//Find associated landmarks
		data_association_per_particle(predicteds, observations,map_landmarks, particle);
		//std::cout << predicteds.id<< predicteds.x<< predicteds.y << std::endl;

		//compute bivariate-gaussian using the link https://en.wikipedia.org/wiki/Multivariate_normal_distribution
		//under section Bivariate case
		double weight_product = 1;
		double sigma_x = std_landmark[0];
		double sigma_y = std_landmark[1];

		for(int j=0; j< observations.size();j++){
			//Compute predicted landmark measurement for the particle
			const LandmarkObs& observation = observations[j];
			const LandmarkObs& predicted = predicteds[j];
			double x = observation.x;
			double y = observation.y;

			double mu_x = predicted.x;
			double mu_y = predicted.y;

			double gap_x = x - mu_x;
			double gap_y = y - mu_y;

			double weight_1 = 1.0/(2 * M_PI * sigma_x * sigma_y);
			double weight_2 = pow(gap_x, 2)/pow(sigma_x, 2) + pow(gap_y,2)/pow(sigma_y, 2);
			weight_2 = exp(-0.5 * weight_2);
			double weight = weight_1 * weight_2;

			//multiply density for all predicted measurements
			weight_product = weight_product * weight;
			
			}
		}

		particle.weight = weight_product;
		weights[i] = weight_product;
	}
}

/*
void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<Particle> particles_backup = particles;
	particles.clear();
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> d( weights.begin(), weights.end());

	for(int i=0; i< num_particles;i++){
		int slected_id = d(gen);
		Particle slected_particle = particles_backup[slected_id];
		particles.push_back(slected_particle);
	}

}
*/

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::default_random_engine gen;
  std::vector<Particle> resampled_particles;
  std::discrete_distribution<int> dist_index(0, num_particles);

  // Calculate the max weight, and setup the distributions
  double max_weight = 0;
  for (int i=0; i<num_particles; i++) {
    max_weight = std::max(particles[i].weight, max_weight);
  }
  std::uniform_real_distribution<double> dist_beta(0, 2.0 * max_weight);

  // Resmaple from particles using the resampling wheel technique described in
  // lesson 13
  int index = dist_index(gen);
  double beta = 0;
  for (int i=0; i<num_particles; i++) {
    beta += dist_beta(gen);
    while (particles[index].weight < beta) {
      beta = beta - particles[index].weight;
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }

  particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
