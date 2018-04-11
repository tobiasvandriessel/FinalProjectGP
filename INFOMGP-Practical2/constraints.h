#ifndef CONSTRAINTS_HEADER_FILE
#define CONSTRAINTS_HEADER_FILE

using namespace Eigen;
using namespace std;

typedef enum ConstraintType { DISTANCE, COLLISION, BARRIER, MIN, MAX, VOLUME, OVERSTRETCH, OVERCOMPR} ConstraintType;   //seems redundant, but you can expand it
typedef enum ConstraintEqualityType { EQUALITY, INEQUALITY } ConstraintEqualityType;

//there is such constraints per two variables that are equal. That is, for every attached vertex there are three such constraints for (x,y,z);
class Constraint {
public:

	VectorXi globalIndices;         //|V_c| list of participating indices (out of global indices of the scene)
	double currValue;               //The current value of the constraint
	RowVectorXd currGradient;       //Current gradient of the constraint
	MatrixXd invMassMatrix;         //M^{-1} matrix size |V_c| X |V_c| only over the participating  vertices
	double refValue;                //Reference values to use in the constraint, when needed
	VectorXd refVector;             //Reference vector when needed
	double CRCoeff;
	ConstraintType constraintType;  //The type of the constraint, and will affect the value and the gradient. This SHOULD NOT change after initialization!
	ConstraintEqualityType constraintEqualityType;  //whether the constraint is an equality or an inequality

	Constraint(const ConstraintType _constraintType, const ConstraintEqualityType _constraintEqualityType, const VectorXi& _globalIndices, const VectorXd& invMasses, const VectorXd& _refVector, const double _refValue, const double _CRCoeff) :constraintType(_constraintType), constraintEqualityType(_constraintEqualityType), refVector(_refVector), refValue(_refValue), CRCoeff(_CRCoeff) {
		currValue = 0.0;
		globalIndices = _globalIndices;
		currGradient = VectorXd::Zero(globalIndices.size());
		invMassMatrix = invMasses.asDiagonal();
	}

	~Constraint() {}


	//updating the value and the gradient vector with given values
	void updateValueGradient(const VectorXd& currVars) {
		switch (constraintType) {

            case OVERSTRETCH: {
                Vector3d v1(currVars(0), currVars(1), currVars(2));
                Vector3d v2(currVars(3), currVars(4), currVars(5));

                currValue =  refValue - (v2 - v1).norm();

                Vector3d gradient = 0.5 * currValue * (v2 - v1) / (v2 - v1).norm();


                currGradient(0) = -gradient(0);
                currGradient(1) = -gradient(1);
                currGradient(2) = -gradient(2);

                currGradient(3) = gradient(0);
                currGradient(4) = gradient(1);
                currGradient(5) = gradient(2);
                break;
            }

            case OVERCOMPR: {
                Vector3d v1(currVars(0), currVars(1), currVars(2));
                Vector3d v2(currVars(3), currVars(4), currVars(5));

                currValue =  (v2 - v1).norm() - refValue;

                Vector3d gradient = -0.5 * currValue * (v2 - v1) / (v2 - v1).norm();


                currGradient(0) = -gradient(0);
                currGradient(1) = -gradient(1);
                currGradient(2) = -gradient(2);

                currGradient(3) = gradient(0);
                currGradient(4) = gradient(1);
                currGradient(5) = gradient(2);
                break;
            }

            case VOLUME: {
                RowVector3d v0(currVars(0), currVars(1), currVars(2));
                RowVector3d v1(currVars(3), currVars(4), currVars(5));
                RowVector3d v2(currVars(6), currVars(7), currVars(8));
                RowVector3d v3(currVars(9), currVars(10), currVars(11));

                RowVector3d e01 = v1 - v0;
			    RowVector3d e02 = v2 - v0;
                RowVector3d e03 = v3 - v0;
        	    double volume = (1.0 / 6.0) * std::abs((e01).dot( (e02).cross(e03) ));
        	    currValue = volume - refValue;

				currGradient(0) = (1 / 6.0)* RowVector3d(-1, 0, 0).dot((e02).cross(e03)) + (1 / 6.0)* (e01).dot((RowVector3d(-1, 0, 0)).cross(e03)) + (1 / 6.0)*(e01).dot((e02).cross(RowVector3d(-1, 0, 0)));
				currGradient(1) = (1 / 6.0)* RowVector3d(0, -1, 0).dot((e02).cross(e03)) + (1 / 6.0)* (e01).dot((RowVector3d(0, -1, 0)).cross(e03)) + (1 / 6.0)*(e01).dot((e02).cross(RowVector3d(0, -1, 0)));
				currGradient(2) = (1 / 6.0)* RowVector3d(0, 0, -1).dot((e02).cross(e03)) + (1 / 6.0)* (e01).dot((RowVector3d(0, 0, -1)).cross(e03)) + (1 / 6.0)*(e01).dot((e02).cross(RowVector3d(0, 0, -1)));
				currGradient(3) = (1 / 6.0)* RowVector3d(1, 0, 0).dot((e02).cross(e03));
				currGradient(4) = (1 / 6.0)* RowVector3d(0, 1, 0).dot((e02).cross(e03));
				currGradient(5) = (1 / 6.0)* RowVector3d(0, 0, 1).dot((e02).cross(e03));
				currGradient(6) = (1 / 6.0)* (e01).dot((RowVector3d(1, 0, 0)).cross(e03));
				currGradient(7) = (1 / 6.0)* (e01).dot((RowVector3d(0, 1, 0)).cross(e03));
				currGradient(8) = (1 / 6.0)* (e01).dot((RowVector3d(0, 0, 1)).cross(e03));
				currGradient(9) = (1 / 6.0)* (e01).dot((e02).cross(RowVector3d(1, 0, 0)));
				currGradient(10) = (1 / 6.0)* (e01).dot((e02).cross(RowVector3d(0, 1, 0)));
				currGradient(11) = (1 / 6.0)* (e01).dot((e02).cross(RowVector3d(0, 0, 1)));
				break;
            }

		case MAX: {
			RowVector3d v1(currVars(0), currVars(1), currVars(2));
			RowVector3d v2(currVars(3), currVars(4), currVars(5));

			currValue = -1. * ((v2 - v1).norm() - refValue);

			RowVector3d gradient =  -1. * ((v2 - v1) / (v2 - v1).norm());


			currGradient(0) = -gradient(0);
			currGradient(1) = -gradient(1);
			currGradient(2) = -gradient(2);

			currGradient(3) = gradient(0);
			currGradient(4) = gradient(1);
			currGradient(5) = gradient(2);


			/***************************
			TODO
			***************************/
			break;
		}

		case MIN: {
			RowVector3d v1(currVars(0), currVars(1), currVars(2));
			RowVector3d v2(currVars(3), currVars(4), currVars(5));

			currValue = (v2 - v1).norm() - refValue;

			RowVector3d gradient = (v2 - v1) / (v2 - v1).norm();


            currGradient(0) = -gradient(0);
            currGradient(1) = -gradient(1);
            currGradient(2) = -gradient(2);

            currGradient(3) = gradient(0);
            currGradient(4) = gradient(1);
			currGradient(5) = gradient(2);


			/***************************
			TODO
			***************************/
			break;
		}
		case DISTANCE: {
			RowVector3d v1(currVars(0), currVars(1), currVars(2));
			RowVector3d v2(currVars(3), currVars(4), currVars(5));

			currValue = (v2 - v1).norm() - refValue;

			RowVector3d gradient = (v2 - v1) / (v2 - v1).norm();


			currGradient(0) = -gradient(0);
			currGradient(1) = -gradient(1);
			currGradient(2) = -gradient(2);

			currGradient(3) = gradient(0);
			currGradient(4) = gradient(1);
			currGradient(5) = gradient(2);


			/***************************
			TODO
			***************************/
			break;
		}

		case COLLISION: {
			/***************************
			TODO
			***************************/
			currValue = refVector.dot(currVars);
			currGradient = refVector;
			break;
		}

		case BARRIER: {
			currValue = currVars(0) - refValue;
			currGradient(0) = 1.0;
			break;
		}
		}
	}


	//computes the impulse needed for all particles to resolve the velocity constraint
	//returns true if constraint was already good
	bool resolveVelocityConstraint(const VectorXd& currPositions, const VectorXd& currVelocities, VectorXd& newImpulses, double tolerance) {

		updateValueGradient(currPositions);

		//		cout << "Curr gradient rows: " << currGradient.rows() << ", cols: " << currGradient.cols() << endl;
		//		cout << "Curr Velocities rows: " << currVelocities.rows() << ", cols: " << currVelocities.cols() << endl;

		if ((constraintEqualityType == INEQUALITY) && ((currValue>-tolerance) || ((currGradient*currVelocities)(0, 0)>-tolerance))) {
			//constraint is valid, or velocity is already solving it, so nothing happens
			newImpulses = VectorXd::Zero(globalIndices.size());
			return true;
		}

		if ((constraintEqualityType == EQUALITY) && (abs((currGradient*currVelocities)(0, 0))<tolerance)) {
			newImpulses = VectorXd::Zero(globalIndices.size());
			return true;
		}

		/***************************
		TODO
		***************************/
		newImpulses = VectorXd::Zero(globalIndices.size());
		double lamda = -currValue / (currGradient * invMassMatrix * currGradient.transpose());
		newImpulses = currGradient.transpose() * lamda;
		return false;
	}

	//projects the position unto the constraint
	//returns true if constraint was already good
	bool resolvePositionConstraint(const VectorXd& currPositions, const VectorXd& currVelocities, VectorXd& newPosDiffs, double tolerance) {

		updateValueGradient(currPositions);
		//cout<<"C(currPositions): "<<currValue<<endl;
		//cout<<"currPositions: "<<currPositions<<endl;
		if ((constraintEqualityType == INEQUALITY) && (currValue>-tolerance)) {
			//constraint is valid
			newPosDiffs = VectorXd::Zero(globalIndices.size());
			return true;
		}

		if ((constraintEqualityType == EQUALITY) && (abs(currValue)<tolerance)) {
			newPosDiffs = VectorXd::Zero(globalIndices.size());
			return true;
		}

		/***************************
		TODO
		***************************/

		newPosDiffs = VectorXd::Zero(globalIndices.size());
		double lamda = -currValue / (currGradient * invMassMatrix * currGradient.transpose());
		newPosDiffs = currGradient.transpose() * invMassMatrix.transpose() * lamda;
		return false;
	}
};



#endif /* constraints_h */
