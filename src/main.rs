mod lib;
use lib::run;

fn main() {

    // Run
    pollster::block_on(run());
}
