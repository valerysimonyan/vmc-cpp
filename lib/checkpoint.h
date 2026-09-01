#pragma once

#include <string> 

#include "physics.h"

void save_checkpoint(const std::string& path, const Ansatz& a);

void load_checkpoint(const std::string& path, Ansatz& a);

void load_transfer(const std::string& path, Ansatz& a);