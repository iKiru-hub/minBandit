#include <iostream>





class MBsolver {

    int K;

public:

    MBsolver(int K) : K(K) {}

    virtual void select_arm() {};
    virtual void update() {};
    virtual void reset() {};
};


class Model : public MBsolver {



};
