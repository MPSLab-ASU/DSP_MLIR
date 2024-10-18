#accesses = [
    affine_map<(m) -> (m)>,
    affine_map<(m) -> (m)>
]

#attrs = {
    indexing_maps = #accesses,
    iterator_types = ["parallel"]
}

func.func @example(%a: memref<?xf32, strided<[1]>>, %b: memref<?xvector<4xf32>, strided<[2], offset: 1>>) {
    linalg.generic #attrs
    ins(%a: memref<?xf32, strided<[1]>>)
    outs(%b: memref<?xvector<4xf32>, strided<[2], offset: 1>>) {
        ^bb0(%aa: f32, %bb:vector<4xf32>):
            %cc = "mk_compute"(%aa, %bb): (f32, vector<4xf32>) -> (vector<4xf32>)
            linalg.yield %cc: vector<4xf32>
    }

    return
}
