function test_basic()
setup_path(); cfg=defaultConfig('quick'); r=validatePhysics(cfg); assert(r.pass);
fprintf('All basic tests passed.\n');
end
